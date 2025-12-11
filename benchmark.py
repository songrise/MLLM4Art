import os

import json
import argparse
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr, combine_pvalues
import sys
import networkx as nx
from typing import List, Tuple, Dict, Any
import pickle
import os

from tqdm import tqdm

#==========stat helpers============

class EloRating:
    def __init__(self, methods, initial_rating=1500, k=32):
        self.k = k
        self.ratings = {method: initial_rating for method in methods}

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, winner, loser):
        rating_w = self.ratings[winner]
        rating_l = self.ratings[loser]

        expected_w = self.expected_score(rating_w, rating_l)
        expected_l = self.expected_score(rating_l, rating_w)

        self.ratings[winner] += self.k * (1 - expected_w)
        self.ratings[loser] += self.k * (0 - expected_l)

    def get_ratings(self):
        return self.ratings


def load_data(json_path):
    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_path}: {e}")
        sys.exit(1)

    df = pd.DataFrame(json_data)
    df = df[df["winner"].isin(["left", "right", None])].reset_index(drop=True)
    methods = pd.unique(df[["method_left", "method_right"]].values.ravel())
    methods = sorted(methods)
    return df, methods


def compute_elo(df, methods):
    elo = EloRating(methods)
    for _, row in df.iterrows():
        if row["winner"] == "left":
            winner = row["method_left"]
            loser = row["method_right"]
        elif row["winner"] == "right":
            winner = row["method_right"]
            loser = row["method_left"]
        else:
            continue  # Skip if winner is not defined
        elo.update_ratings(winner, loser)
    elo_ratings = elo.get_ratings()
    elo_ranking = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    return elo_ranking, elo_ratings


def compute_bradley_terry(df, methods):
    pairwise = pd.DataFrame(0, index=methods, columns=methods, dtype=int)
    for _, row in df.iterrows():
        if row["winner"] == "left":
            winner = row["method_left"]
            loser = row["method_right"]
        elif row["winner"] == "right":
            winner = row["method_right"]
            loser = row["method_left"]
        else:
            continue  # Skip if winner is not defined
        pairwise.loc[winner, loser] += 1

    def negative_log_likelihood(theta, pairwise_matrix):
        n = len(theta)
        ll = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w_ij = pairwise_matrix.iloc[i, j]
                if w_ij > 0:
                    ll += -w_ij * (
                        theta[i] - np.log(np.exp(theta[i]) + np.exp(theta[j]))
                    )
        return ll

    initial_theta = np.zeros(len(methods))

    constraints = {"type": "eq", "fun": lambda theta: np.sum(theta)}

    result = minimize(
        negative_log_likelihood,
        initial_theta,
        args=(pairwise,),
        method="SLSQP",
        constraints=constraints,
    )

    if not result.success:
        print("Optimization for Bradley-Terry Model failed:", result.message)
        sys.exit(1)

    theta = result.x
    bt_scores = {method: score for method, score in zip(methods, theta)}
    bt_ranking = sorted(bt_scores.items(), key=lambda x: x[1], reverse=True)
    return bt_ranking, bt_scores


def spearman_correlation(ranking1, ranking2, methods, metric_name):
    # Create dictionaries for quick lookup
    rank_dict1 = {method: rank for rank, (method, _) in enumerate(ranking1, start=1)}
    rank_dict2 = {method: rank for rank, (method, _) in enumerate(ranking2, start=1)}

    # Ensure both rankings have the same methods
    common_methods = set(rank_dict1.keys()) & set(rank_dict2.keys())
    if len(common_methods) == 0:
        print(f"No common methods to compare for {metric_name}.")
        return None, None

    ranks1 = [rank_dict1[m] for m in common_methods]
    ranks2 = [rank_dict2[m] for m in common_methods]

    rho, pval = spearmanr(ranks1, ranks2)
    return rho, pval


def print_ranking(title, ranking, score_format=".2f"):
    print(f"\n{title}:")
    for rank, (method, score) in enumerate(ranking, start=1):
        print(f"{rank}. {method}: {score:{score_format}}")


def random_split(max_idx, n_split):
    """
    Randomly split a list of indices into n_split groups.

    :param max_idx: Maximum index to split.
    :param n_split: Number of groups to split into.
    :return: List of lists containing the indices for each group.
    """
    #fix seed
    np.random.seed(8)
    indices = list(range(max_idx))
    np.random.shuffle(indices)
    return [indices[i::n_split] for i in range(n_split)]



def preprocess_instance_df(
    df: pd.DataFrame, inconsistency_threshold: float = 0.15
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Preprocess the DataFrame by:
    1. Removing skipped instances.
    2. Removing groups with fewer than 15 non-skipped instances.
    3. Removing highly inconsistent groups based on the Feedback Arc Set (FAS) algorithm.

    All removals are performed collectively at the end to ensure consistency.

    :param df: pandas DataFrame containing the instances.
    :param inconsistency_threshold: Threshold proportion of edges to remove for inconsistency.
                                    Groups exceeding this threshold will be dropped.
    :return: Tuple containing the cleaned DataFrame and list of removed global DataFrame indices.
    """

    ids_to_remove: Set[int] = set()  # Use a set to avoid duplicate indices

    # Step 1: Identify Skipped Instances
    if "skipped" not in df.columns:
        # conventional metrics, return as is
        print("No 'skipped' column found. Returning original DataFrame.")
        return df, []

    skipped_df = df[df["skipped"] == True]
    skipped_ids = skipped_df.index.tolist()
    ids_to_remove.update(skipped_ids)

    print(f"Identified {len(skipped_ids)} skipped instances to remove.")
    percentage_skipped = (len(skipped_ids) / len(df)) * 100 if len(df) > 0 else 0
    print(f"Skipped instances represent {percentage_skipped:.2f}% of the total data.")

    # Step 2: Group by Each Unique Instance (content_index & style_index)
    grouped = df.groupby(["content_index", "style_index"])

    # List to keep track of groups marked for removal
    groups_marked_for_removal = []

    # Iterate through each group (unique instance)
    for (content_idx, style_idx), group in grouped:
        # Separate skipped and non-skipped rows within the group
        non_skipped_group = group[group["skipped"] == False]

        # Condition 1: Drop groups with fewer than 15 non-skipped instances
        if len(non_skipped_group) < 15:
            groups_marked_for_removal.append((content_idx, style_idx))
            continue

        # Condition 2: Apply Feedback Arc Set (FAS) to detect inconsistency on non-skipped rows
        G = nx.DiGraph()

        # Add edges based on preferences
        for _, row in non_skipped_group.iterrows():
            method_left = row["method_left"]
            method_right = row["method_right"]
            winner = row["winner"]

            if winner is None:
                # If winner is None, skip this preference
                continue

            winner = winner.lower()

            if winner == "left":
                source, target = method_left, method_right
            elif winner == "right":
                source, target = method_right, method_left
            else:
                # Invalid winner value; skip this preference
                continue

            G.add_edge(source, target)

        # Apply Feedback Arc Set (FAS) to detect cycles
        fas_edges = approximate_feedback_arc_set(G)

        total_edges = G.number_of_edges()
        edges_to_remove = len(fas_edges)
        inconsistency_ratio = edges_to_remove / total_edges if total_edges > 0 else 0

        print(
            f"Group (Content: {content_idx}, Style: {style_idx}) - Total Edges: {total_edges}, "
            f"Edges to Remove: {edges_to_remove}, Inconsistency Ratio: {inconsistency_ratio:.2f}"
        )

        # If inconsistency_ratio exceeds the threshold, mark the group for removal
        if inconsistency_ratio > inconsistency_threshold:
            groups_marked_for_removal.append((content_idx, style_idx))

    # Step 3: Collect Indices from Marked Groups
    if groups_marked_for_removal:
        # Filter DataFrame for groups to remove
        mask = df.apply(
            lambda row: (row["content_index"], row["style_index"])
            in groups_marked_for_removal,
            axis=1,
        )
        inconsistent_df = df[mask]
        inconsistent_ids = inconsistent_df.index.tolist()
        ids_to_remove.update(inconsistent_ids)

        print(
            f"Identified {len(inconsistent_ids)} inconsistent instances from {len(groups_marked_for_removal)} groups to remove."
        )
    else:
        print("No inconsistent groups found exceeding the threshold.")

    # Convert the set of indices to a sorted list for consistency
    ids_to_remove = sorted(ids_to_remove)

    # Step 4: Drop All Identified Indices at Once
    df_final = df.drop(ids_to_remove).reset_index(drop=True)

    # Calculate and display the percentage of data removed
    total_removed = len(ids_to_remove)
    percentage_removed = (total_removed / len(df)) * 100 if len(df) > 0 else 0
    print(
        f"Total instances removed: {total_removed} ({percentage_removed:.2f}% of the data)."
    )

    return df_final, ids_to_remove


def approximate_feedback_arc_set(G: nx.DiGraph) -> List[Tuple[Any, Any]]:
    """
    Approximate the Feedback Arc Set of a directed graph using a greedy heuristic.

    :param G: A directed graph (networkx.DiGraph).
    :return: List of edges that form the approximate Feedback Arc Set.
    """
    fas = []
    G_copy = G.copy()

    # Calculate scores for each node: out_degree - in_degree
    scores = {
        node: G_copy.out_degree(node) - G_copy.in_degree(node)
        for node in G_copy.nodes()
    }

    # Order nodes based on scores
    ordered_nodes = sorted(scores, key=lambda x: scores[x], reverse=True)

    # Create a ranking dictionary
    ranking = {node: idx for idx, node in enumerate(ordered_nodes)}

    # Identify edges that go against the ranking
    for u, v in G_copy.edges():
        if ranking[u] > ranking[v]:
            fas.append((u, v))

    return fas

def main(human_path, vlm_path, mode, args):
    # Load data
    human_df, human_methods = load_data(human_path)
    vlm_df, vlm_methods = load_data(vlm_path)
    # Truncate the data
    human_df = human_df[: args.before]
    vlm_df = vlm_df[: args.before]

    # Ensure the methods are consistent across both datasets
    all_methods = sorted(list(set(human_methods) | set(vlm_methods)))

    if mode == "global":
        # Compute ELO rankings
        if args.n_split > 1:
            splits = random_split(len(vlm_df), args.n_split)
        else:
            splits = [list(range(len(human_df)))]
        
        if args.n_split > 1:
            all_elo_rho, all_elo_p = [], []
            all_bt_rho, all_bt_p = [], []
            for s_index, split in enumerate(splits):
                human_df_split = human_df.iloc[split]
                vlm_df_split = vlm_df.iloc[split]
                print("="*60)
                print(f"Stats for split {s_index}...")

                # Compute ELO rankings
                human_elo_ranking, human_elo_scores = compute_elo(human_df_split, all_methods)
                vlm_elo_ranking, vlm_elo_scores = compute_elo(vlm_df_split, all_methods)

                # Compute Bradley-Terry rankings
                human_bt_ranking, human_bt_scores = compute_bradley_terry(human_df_split, all_methods)
                vlm_bt_ranking, vlm_bt_scores = compute_bradley_terry(vlm_df_split, all_methods)
                # Print Rankings
                print_ranking("Human ELO Rankings", human_elo_ranking)
                print_ranking(
                    "Human Bradley-Terry Rankings", human_bt_ranking, score_format=".4f"
                )
                print_ranking("VLM ELO Rankings", vlm_elo_ranking)
                print_ranking("VLM Bradley-Terry Rankings", vlm_bt_ranking, score_format=".4f")
                # Correlation Analysis
                elo_rho, elo_p = spearman_correlation(
                    human_elo_ranking, vlm_elo_ranking, all_methods, "ELO"
                )
                bt_rho, bt_p = spearman_correlation(
                    human_bt_ranking, vlm_bt_ranking, all_methods, "Bradley-Terry"
                )
                all_elo_rho.append(elo_rho)
                all_elo_p.append(elo_p)
                all_bt_rho.append(bt_rho)
                all_bt_p.append(bt_p)

            elo_combined_rho = np.mean(all_elo_rho)
            bt_combined_rho = np.mean(all_bt_rho)

            #combine pvalues
            elo_combined_p = combine_pvalues(all_elo_p, method=args.p_combine)[1]
            bt_combined_p = combine_pvalues(all_bt_p, method=args.p_combine)[1]
            # print the rho and p for each split
            print("="*60)
            print("Correlation Analysis (Per Split):\n")
            for s_index in range(args.n_split):
                print(f"Split {s_index}, ELO Rankings Spearman's rho: {all_elo_rho[s_index]:.4f}, p-value: {all_elo_p[s_index]:.4e}")
                print(f"Split {s_index}, Bradley-Terry Rankings Spearman's rho: {all_bt_rho[s_index]:.4f}, p-value: {all_bt_p[s_index]:.4e}") 
            # Print Correlation Results
            print("="*60)
            print("Correlation Analysis Summary (Aggregated):\n")
            if elo_combined_rho is not None:
                print(f"{args.n_split} Split, ELO Rankings Combined Spearman's rho: {elo_combined_rho:.4f}, Combined p-value: { elo_combined_p:.4e}")
            if bt_combined_rho is not None:
                print(
                    f"{args.n_split} Split, Bradley-Terry Rankings Combined Spearman's rho: {bt_combined_rho:.4f}, Combined p-value: {bt_combined_p:.4e}"
                )

        else: # single split for global analysis, the p-value may not be very reliable.
                    
            human_elo_ranking, human_elo_scores = compute_elo(human_df, all_methods)
            vlm_elo_ranking, vlm_elo_scores = compute_elo(vlm_df, all_methods)

            # Compute Bradley-Terry rankings
            human_bt_ranking, human_bt_scores = compute_bradley_terry(human_df, all_methods)
            vlm_bt_ranking, vlm_bt_scores = compute_bradley_terry(vlm_df, all_methods)

            # Print Rankings
            print_ranking("Human ELO Rankings", human_elo_ranking)
            print_ranking(
                "Human Bradley-Terry Rankings", human_bt_ranking, score_format=".4f"
            )
            print_ranking("VLM ELO Rankings", vlm_elo_ranking)
            print_ranking("VLM Bradley-Terry Rankings", vlm_bt_ranking, score_format=".4f")

            # Correlation Analysis
            elo_rho, elo_p = spearman_correlation(
                human_elo_ranking, vlm_elo_ranking, all_methods, "ELO"
            )
            bt_rho, bt_p = spearman_correlation(
                human_bt_ranking, vlm_bt_ranking, all_methods, "Bradley-Terry"
            )

            # Print Correlation Results
            print("="*60)
            print("Correlation Analysis:")
            if elo_rho is not None:
                print(f"ELO Rankings Spearman's rho: {elo_rho:.4f}, p-value: {elo_p:.4e}")
            if bt_rho is not None:
                print(
                    f"Bradley-Terry Rankings Spearman's rho: {bt_rho:.4f}, p-value: {bt_p:.4e}"
                )
            # detect mis-aligned 2AFC and save as json.
            misaligned = []
            for i, row in human_df.iterrows():
                content_idx = row["content_index"]
                style_idx = row["style_index"]
                human_winner = row["winner"]
                try:
                    vlm_winner = vlm_df.iloc[i]["winner"]
                    # get the vlm response words
                    vlm_response = vlm_df.iloc[i]["conversation"]
                    if human_winner != vlm_winner:
                        misaligned.append(
                            {
                                "id" : row["id"],
                                "content_index": content_idx,
                                "style_index": style_idx,
                                "style_prompt": row["style_prompt"],
                                "human_winner": human_winner,
                                "vlm_winner": vlm_winner,
                                "vlm_response": vlm_response,
                            }
                        )
                except Exception as e:
                    continue
        
            exp_name = os.path.basename(vlm_path).split(".")[0]
            out_misaligned = os.path.join(
                f"./out/misaligned_{exp_name}.json"
            )
            with open(out_misaligned, "w") as f:
                json.dump(misaligned, f, indent=4)
            print(
                f"Misaligned instances saved to {out_misaligned}. Total: {len(misaligned)}"
            )
    elif mode == "instance":
        group_corr = {}
        # List of the indices for the same content and style
        human_df, ids_to_remove = preprocess_instance_df(human_df)
        ids_to_remove = list(sorted(ids_to_remove))
        print(f"Removed {len(ids_to_remove)} instances from human annotations.")

        #output the filtered instance ids so that API query can skip them to save money.
        pickle.dump(
            ids_to_remove,
            open("./out/ids_to_remove.pkl", "wb"),
        )
     
        # Compute  rankings for each instance grouped by content and style
        elo_groups_human, bt_groups_human = {}, {}
        elo_groups_vlm, bt_groups_vlm = {}, {}


        # Correlation Analysis for rank-level
        print("Computing ELO and Bradley-Terry rankings for each instance...")

        for (content_idx, style_idx), group in human_df.groupby(
            ["content_index", "style_index"]
        ):
            elo_ranking, _ = compute_elo(group, all_methods)
            bt_ranking, _ = compute_bradley_terry(group, all_methods)
            elo_groups_human[(content_idx, style_idx)] = elo_ranking
            bt_groups_human[(content_idx, style_idx)] = bt_ranking

        for (content_idx, style_idx), group in vlm_df.groupby(
            ["content_index", "style_index"]
        ):
            elo_ranking, _ = compute_elo(group, all_methods)
            bt_ranking, _ = compute_bradley_terry(group, all_methods)
            elo_groups_vlm[(content_idx, style_idx)] = elo_ranking
            bt_groups_vlm[(content_idx, style_idx)] = bt_ranking

        # Compute the correlation between human and VLM rankings for each instance
        elo_correlations, elo_ps = [], []
        bt_correlations, bt_ps = [], []
        for key in elo_groups_human.keys():
            if key in elo_groups_vlm:
                elo_rho, elo_p = spearman_correlation(
                    elo_groups_human[key], elo_groups_vlm[key], all_methods, "ELO"
                )
                bt_rho, bt_p = spearman_correlation(
                    bt_groups_human[key],
                    bt_groups_vlm[key],
                    all_methods,
                    "Bradley-Terry",
                )
                elo_correlations.append(elo_rho)
                bt_correlations.append(bt_rho)
                elo_ps.append(elo_p)
                bt_ps.append(bt_p)
                group_corr[key] = {
                    "elo_rho": elo_rho,
                    "elo_p": elo_p,
                    "bt_rho": bt_rho,
                    "bt_p": bt_p,
                }
        # Print the average correlation and p-value
        elo_combined_ps = combine_pvalues(elo_ps, method=args.p_combine)[1]
        bt_combined_ps = combine_pvalues(bt_ps, method=args.p_combine)[1]
        print(
            f"ELO Rankings Spearman's rho: {np.mean(elo_correlations):.4f}, p-value (fisher): {elo_combined_ps:.4e}"
        )
        print(
            f"Bradley-Terry Rankings Spearman's rho: {np.mean(bt_correlations):.4f}, p-value (fisher): {bt_combined_ps:.4e}"
        )


        # detect mis-aligned 2AFC and save as json.
        misaligned = []
        for i, row in human_df.iterrows():
            content_idx = row["content_index"]
            style_idx = row["style_index"]
            human_winner = row["winner"]
            try:
                vlm_winner = vlm_df.iloc[i]["winner"]
                # get the vlm response words
                vlm_response = vlm_df.iloc[i]["conversation"]
                if human_winner != vlm_winner:
                    misaligned.append(
                        {
                            "id" : row["id"],
                            "content_index": content_idx,
                            "style_index": style_idx,
                            "human_winner": human_winner,
                            "vlm_winner": vlm_winner,
                            "vlm_response": vlm_response,
                        }
                    )
            except Exception as e:
                continue
        
        exp_name = os.path.basename(vlm_path).split(".")[0]
        out_misaligned = os.path.join(
            f"./out/misaligned_{exp_name}.json"
        )
        with open(out_misaligned, "w") as f:
            json.dump(misaligned, f, indent=4)
        print(
            f"Misaligned instances saved to {out_misaligned}. Total: {len(misaligned)}"
        )
    else:
        print(f"Unknown mode: {mode}. Please choose 'global' or 'instance'.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ELO and Bradley-Terry rankings for human and VLM annotations."
    )
    parser.add_argument(
        "--human_annotation", "--ha",
        type=str,
        default="./data/human_annotation/global_annotation.json",
        help="Path to human annotations JSON file.",
    )
    parser.add_argument(
        "--model_annotation", "--ma", type=str, required=True, help="Path to model annotations JSON file."
    )
    parser.add_argument(
        "--before", type=int, default=1000, help="calculate the data before idx."
    )
    parser.add_argument("--n_split", "--ns", type=int, default=5, help="Number of independent and random splits for improving robustness of correlation analysis in global (per-artist) mode. Not used in instance mode.")
    parser.add_argument("--mode", choices = ["global", "instance"], type=str, default="global")
    parser.add_argument("--p_combine", type=str, default="fisher", help="p-value combination method")
    args = parser.parse_args()

    main(args.human_annotation, args.model_annotation, args.mode, args)

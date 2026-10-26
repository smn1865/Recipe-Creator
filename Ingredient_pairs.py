"""Find which ingredients go together, based on real recipes.

Step 1 of the recipe algorithm. For every pair of ingredients it measures
how often they appear in the same recipe, compared with pure chance.

Key score — LIFT:
    lift = P(A and B together) / (P(A) * P(B))
    lift > 1  -> they appear together MORE than chance (good pairing)
    lift = 1  -> no relationship
    lift < 1  -> they tend to avoid each other

Run from the repository root (needs recipes.csv there):
    python3 ingredient_pairs.py
Output: ingredient_pairs.csv (ignored by git, like all .csv files)
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

from clean_recipes import clean_text, enforce_id_integrity, handle_missing, load_recipes

MIN_INGREDIENT_RECIPES = 50  # ignore ingredients used in fewer recipes (typos, rarities)
MIN_PAIR_RECIPES = 20        # ignore pairs seen too rarely (lift is unreliable for rare pairs)
OUTPUT_PATH = Path("ingredient_pairs.csv")


def parse_ingredients(raw: str) -> list[str]:
    """Turn 'c("butter", "Sugar")' into ['butter', 'sugar'] (unique, lowercase)."""
    items = re.findall(r'"([^"]*)"', raw)
    return sorted({i.strip().lower() for i in items if i.strip()})


def build_pair_table(ingredient_lists: pd.Series) -> pd.DataFrame:
    n_recipes = len(ingredient_lists)

    # 1. How many recipes use each ingredient
    single = Counter(i for items in ingredient_lists for i in items)
    common = {i for i, c in single.items() if c >= MIN_INGREDIENT_RECIPES}

    # 2. How many recipes use each pair
    pairs: Counter = Counter()
    for items in ingredient_lists:
        kept = [i for i in items if i in common]
        pairs.update(combinations(kept, 2))

    # 3. Scores
    rows = []
    for (a, b), together in pairs.items():
        if together < MIN_PAIR_RECIPES:
            continue
        lift = (together / n_recipes) / ((single[a] / n_recipes) * (single[b] / n_recipes))
        rows.append({
            "ingredient_a": a,
            "ingredient_b": b,
            "recipes_together": together,
            "recipes_with_a": single[a],
            "recipes_with_b": single[b],
            "pct_of_a_recipes_with_b": round(100 * together / single[a], 1),
            "lift": round(lift, 2),
        })
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def best_partners(pair_table: pd.DataFrame, ingredient: str, n: int = 10) -> pd.DataFrame:
    """Top-n ingredients that pair best with one ingredient."""
    ingredient = ingredient.lower()
    mask = (pair_table["ingredient_a"] == ingredient) | (pair_table["ingredient_b"] == ingredient)
    hits = pair_table[mask].copy()
    hits["partner"] = hits["ingredient_b"].where(hits["ingredient_a"] == ingredient, hits["ingredient_a"])
    return hits[["partner", "recipes_together", "lift"]].head(n)


def main() -> None:
    recipes = load_recipes()
    recipes = handle_missing(recipes)
    recipes = clean_text(recipes)
    recipes = enforce_id_integrity(recipes)

    ingredient_lists = recipes["RecipeIngredientParts"].apply(parse_ingredients)
    ingredient_lists = ingredient_lists[ingredient_lists.str.len() >= 2]
    print(f"\nRecipes with 2+ ingredients: {len(ingredient_lists):,}")

    pair_table = build_pair_table(ingredient_lists)
    pair_table.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(pair_table):,} ingredient pairs to {OUTPUT_PATH}")

    print("\nStrongest pairings overall:")
    print(pair_table.head(15).to_string(index=False))

    for example in ["chicken", "chocolate", "tomatoes"]:
        print(f"\nBest partners for '{example}':")
        print(best_partners(pair_table, example).to_string(index=False))


if __name__ == "__main__":
    main()
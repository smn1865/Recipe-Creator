# Esli shto stei commentnery chjnjeq indz petqa

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_COLUMNS = [
    "RecipeId",
    "Name",
    "RecipeCategory",
    "RecipeIngredientParts",
]
TEXT_COLUMNS = ["Name", "RecipeCategory", "RecipeIngredientParts"]
DEFAULT_CSV_PATH = Path("recipes.csv")


def load_recipes(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load only the four target columns."""
    recipes = pd.read_csv(csv_path, usecols=TARGET_COLUMNS)
    print("Initial shape:", recipes.shape)
    print()
    recipes.info()
    return recipes


def report_missing(recipes: pd.DataFrame, label: str = "Missing values per column") -> None:
    print(f"\n{label}:")
    print(recipes.isna().sum())


def handle_missing(recipes: pd.DataFrame) -> pd.DataFrame:
    """Drop missing RecipeId rows; fill text NaNs with safe defaults."""
    recipes = recipes.copy()

    missing_ids = int(recipes["RecipeId"].isna().sum())
    if missing_ids:
        recipes = recipes.dropna(subset=["RecipeId"])
        print(f"Dropped {missing_ids} row(s) with missing RecipeId.")
    else:
        print("\nNo missing RecipeId values.")

    recipes["Name"] = recipes["Name"].fillna("")
    recipes["RecipeIngredientParts"] = recipes["RecipeIngredientParts"].fillna("")
    recipes["RecipeCategory"] = recipes["RecipeCategory"].fillna("Unknown")
    return recipes


def clean_text(recipes: pd.DataFrame) -> pd.DataFrame:
    """Strip, collapse whitespace, and title-case categories (vectorised)."""
    recipes = recipes.copy()
    for col in TEXT_COLUMNS:
        recipes[col] = (
            recipes[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    recipes["RecipeCategory"] = recipes["RecipeCategory"].str.title()
    return recipes


def enforce_id_integrity(recipes: pd.DataFrame) -> pd.DataFrame:
    """Cast RecipeId to int64 and drop duplicate IDs."""
    recipes = recipes.copy()
    recipes["RecipeId"] = pd.to_numeric(recipes["RecipeId"], errors="coerce")
    invalid_ids = int(recipes["RecipeId"].isna().sum())
    if invalid_ids:
        recipes = recipes.dropna(subset=["RecipeId"])
        print(f"Dropped {invalid_ids} row(s) with invalid RecipeId.")

    recipes["RecipeId"] = recipes["RecipeId"].astype(np.int64)

    duplicate_count = int(recipes["RecipeId"].duplicated().sum())
    recipes = recipes.drop_duplicates(subset=["RecipeId"], keep="first")
    if duplicate_count:
        print(f"Removed {duplicate_count} duplicate RecipeId row(s).")
    else:
        print("No duplicate RecipeId values.")
    return recipes.reset_index(drop=True)


def plot_top_categories(recipes: pd.DataFrame, n: int = 10) -> None:
    """Bar chart of the n most frequent RecipeCategory values."""
    top_n = recipes["RecipeCategory"].value_counts().head(n)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top_n.index.astype(str), top_n.to_numpy())
    ax.set_title(f"Top {n} Recipe Categories")
    ax.set_xlabel("RecipeCategory")
    ax.set_ylabel("Number of recipes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if plt.isinteractive() or plt.get_backend().lower() != "agg":
        plt.show()


def inspect_cleaned(recipes: pd.DataFrame) -> None:
    print("\nFinal shape:", recipes.shape)
    print("\nRemaining missing values:")
    print(recipes.isna().sum())
    print("\nDtypes:")
    print(recipes.dtypes)
    print("\nFirst 10 cleaned rows:")
    with pd.option_context("display.max_columns", None, "display.max_colwidth", 80, "display.width", 160):
        print(recipes.head(10))


# First matching regex wins. Keep specific dish types ahead of leftovers / "Other".
PARENT_CATEGORY_RULES: list[tuple[str, str]] = [
    (r"soup|stew|chowder|gumbo|\bstocks?\b", "Soups & Stews"),
    (r"chicken|poultry|\bturkey\b|\bduck\b|goose|\bquail\b|pheasant", "Chicken"),
    (r"savory pies?|pot pie", "Main Dishes"),
    (
        r"dessert|cookie|cheesecake|ice cream|candy|gelatin|\bcake\b|tarts?|pudding|fudge",
        "Desserts",
    ),
    (r"\bpie\b", "Desserts"),
    (r"^sweet$|snacks sweet", "Desserts"),
    (
        r"beef|steak|pork|\bham\b|lamb|veal|\bmeats?\b|meatball|meatloaf|"
        r"wild game|\bdeer\b|\belk\b|\bmoose\b|\bbear\b|\brabbit\b",
        "Meat",
    ),
    (
        r"fish|salmon|tuna|crab|lobster|mussel|oyster|squid|octopus|catfish|"
        r"\bbass\b|trout|tilapia|halibut|mahi|perch|crawfish|whitefish|orange roughy",
        "Seafood",
    ),
    (r"breakfast|brunch|oatmeal", "Breakfast"),
    (r"breads?|scones?|biscuits?|\bbaking\b", "Breads"),
    (r"beverage|shakes?|smoothie|punch|margarita", "Beverages"),
    (r"sauces?|dressing|spreads?|chutney|gravy|\bdip\b", "Sauces & Condiments"),
    (r"pasta|spaghetti|penne|macaroni|manicotti|\brice\b|grains?", "Pasta & Grains"),
    (
        r"vegetable|potato|spinach|\bcorn\b|pepper|\bonions?\b|\bbeans?\b|lentil|"
        r"cauliflower|broccoli|avocado|greens?|\bchard\b|\byam\b|artichoke|"
        r"soy/tofu|tempeh|\bfruit|\bberries\b|\bapple\b|\boranges?\b|\blemon\b|"
        r"\blime\b|mango|melon|\bpears?\b|\bplums?\b|cherries|grapes?|pineapple|"
        r"coconut|citrus|papaya|kiwi|pumpkin|strawberry",
        "Vegetables & Produce",
    ),
    (r"snack|appetizer|lunch", "Snacks & Lunch"),
    (r"one dish|casserole|pot roast|\broast\b|stir fry", "Main Dishes"),
    (r"\bcheese\b", "Dairy"),
]


def consolidate_recipe_category(category: pd.Series) -> pd.Series:
    """Map fragmented RecipeCategory labels to ~15 parent categories."""
    lower = category.fillna("").astype(str).str.strip().str.lower()
    parent = pd.Series("Other", index=category.index, dtype="object")
    assigned = pd.Series(False, index=category.index)

    for pattern, label in PARENT_CATEGORY_RULES:
        hit = ~assigned & lower.str.contains(pattern, regex=True, na=False)
        parent[hit] = label
        assigned |= hit

    return parent


def deep_clean_recipes(recipes: pd.DataFrame) -> pd.DataFrame:
    """Final cleanup: short names, conceptual duplicates, category rollup."""
    before_shape = recipes.shape
    print("\nDeep clean")
    print("Shape before:", before_shape)

    cleaned = recipes.copy()
    name_len = cleaned["Name"].fillna("").astype(str).str.len()
    cleaned = cleaned.loc[name_len >= 3]

    cleaned = cleaned.drop_duplicates(
        subset=["Name", "RecipeIngredientParts"],
        keep="first",
    )

    cleaned["RecipeCategory"] = consolidate_recipe_category(cleaned["RecipeCategory"])
    cleaned = cleaned.reset_index(drop=True)

    print("Shape after:", cleaned.shape)
    print("\nConsolidated RecipeCategory value counts:")
    print(cleaned["RecipeCategory"].value_counts().to_string())
    print(f"\nUnique parent categories: {cleaned['RecipeCategory'].nunique()}")
    return cleaned


def clean_recipes(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    recipes = load_recipes(csv_path)
    report_missing(recipes)
    recipes = handle_missing(recipes)
    recipes = clean_text(recipes)
    recipes = enforce_id_integrity(recipes)
    inspect_cleaned(recipes)
    plot_top_categories(recipes, n=10)
    return recipes


if __name__ == "__main__":
    clean_recipes()

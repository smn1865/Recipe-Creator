# Stage 1 - Workstream A: data cleaning for Food.com recipes.csv
# Put recipes.csv next to this file and run:  python data_cleaning.py
#
# Rules we follow:
#  - the raw CSV is never changed
#  - we do not silently delete rows: every row gets flags, and rows that are
#    left out of the clean set get an "exclusion_reason"

import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "recipes.csv"
# Output files are saved in the same folder as the script.
# To use a folder instead, create it first and set OUT_DIR = "output/"
OUT_DIR = ""
# If True, recipes with the same name AND the same ingredients are kept only once
# in the clean set (so Apriori support counts are not inflated). They are still
# listed in recipes_flagged.csv, so nothing is lost.
DROP_CONCEPTUAL_DUPLICATES = True

# ---------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------
cols = ["RecipeId", "Name", "RecipeCategory", "RecipeIngredientParts"]
df = pd.read_csv(CSV_PATH, usecols=cols)
df["source_row"] = df.index  # so we can always find the original row

print("Shape:", df.shape)
print(df.info())
print("\nMissing values:")
print(df.isna().sum())

# ---------------------------------------------------------------
# 2. Validate RecipeId (no silent truncation, no dropping)
# ---------------------------------------------------------------
df["RecipeId"] = pd.to_numeric(df["RecipeId"], errors="coerce")

# valid = a number, a whole number (12.7 is not valid), and positive
df["id_ok"] = df["RecipeId"].notna() & (df["RecipeId"] % 1 == 0) & (df["RecipeId"] > 0)
print("\nInvalid RecipeIds:", (~df["id_ok"]).sum())

# ---------------------------------------------------------------
# 3. Duplicate RecipeIds: exact copies vs conflicting records
# ---------------------------------------------------------------
# How many different versions of each RecipeId exist?
unique_versions = df.drop_duplicates(subset=cols)
versions_per_id = unique_versions.groupby("RecipeId").size()
df["id_versions"] = df["RecipeId"].map(versions_per_id)

# conflicting = same id but the rows are not identical
df["id_conflict"] = df["id_versions"] > 1
# extra exact copy = identical row that already appeared earlier
df["extra_exact_copy"] = df.duplicated(subset=cols, keep="first") & ~df["id_conflict"]

print("Rows with conflicting duplicate IDs:", df["id_conflict"].sum())
print("Extra exact duplicate rows:", df["extra_exact_copy"].sum())

# ---------------------------------------------------------------
# 4. Text columns (whitespace only, we keep the raw category)
# ---------------------------------------------------------------
df["Name"] = df["Name"].str.replace(r"\s+", " ", regex=True).str.strip()
df["name_missing"] = df["Name"].isna() | (df["Name"] == "")
df["name_short"] = df["Name"].str.len() < 3  # flag only, we keep these rows

df["category_key"] = (
    df["RecipeCategory"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.lower()
    .replace("", pd.NA)
)
df["category_missing"] = df["category_key"].isna()
df["category_key"] = df["category_key"].fillna("unknown")


# ---------------------------------------------------------------
# 5. Parse and normalize RecipeIngredientParts
# ---------------------------------------------------------------
# In recipes.csv the list columns are written in R format, for example:
#   c("chicken breast", "garlic", "rice")     or     character(0)
# so we cannot use ast.literal_eval. We use simple string methods instead.

def parse_ingredients(text):
    """Turn the raw string into a Python list. Returns None if the format is unknown."""
    if pd.isna(text):
        return None
    text = text.strip()
    if text == "character(0)":
        return []
    if text.startswith("c(") and text.endswith(")"):
        inside = text[2:-1].strip()  # remove the c( and the )
        if inside == "":
            return []
        if inside.startswith('"') and inside.endswith('"'):
            inside = inside[1:-1]  # remove the first and last quote
            return inside.split('", "')  # items are separated by ", "
        return None  # unknown format -> we flag it, we do not guess
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        return [text[1:-1]]  # a recipe with only one ingredient
    return None  # unknown format -> we flag it, we do not guess


def clean_ingredients(items):
    """Lowercase, trim spaces, drop empty items and duplicates (keep the order)."""
    if items is None:
        return None
    result = []
    for item in items:
        item = item.replace('\\"', '"')  # R writes a quote inside a name as \"
        item = " ".join(item.split()).lower()  # fix extra spaces, then lowercase
        if item != "" and item not in result:
            result.append(item)
    return result


df["ingredients_raw_list"] = df["RecipeIngredientParts"].apply(parse_ingredients)
df["ingredients"] = df["ingredients_raw_list"].apply(clean_ingredients)

df["ingredients_ok"] = df["ingredients"].notna()
df["n_ingredients"] = df["ingredients"].apply(lambda x: len(x) if x is not None else 0)
df["had_ingredient_dups"] = df["ingredients_raw_list"].apply(
    lambda x: x is not None and len(x) != len(set(i.strip().lower() for i in x))
)
# we save ingredient lists as text joined with "|", so make sure no name has a "|"
df["has_pipe"] = df["ingredients"].apply(
    lambda x: x is not None and any("|" in i for i in x)
)

print("\nIngredient lists in an unknown format:", (~df["ingredients_ok"]).sum())
print("Recipes with 0 ingredients:", (df["n_ingredients"] == 0).sum())
print("Ingredient names containing '|':", df["has_pipe"].sum())

# ---------------------------------------------------------------
# 6. Decide which rows go into the clean set (and write down why not)
# ---------------------------------------------------------------
df["exclusion_reason"] = ""


def set_reason(mask, reason):
    """Only fill the reason if the row has no reason yet (first reason wins)."""
    df.loc[mask & (df["exclusion_reason"] == ""), "exclusion_reason"] = reason


set_reason(~df["id_ok"], "invalid_id")
set_reason(df["id_conflict"], "conflicting_duplicate_id")
set_reason(df["extra_exact_copy"], "exact_duplicate_id_extra_copy")
set_reason(~df["ingredients_ok"], "ingredients_unknown_format")
set_reason(df["n_ingredients"] == 0, "no_ingredients")

# ---------------------------------------------------------------
# 7. Conceptual duplicates: same name + same ingredients
# ---------------------------------------------------------------
# These are only FLAGGED (different recipes can look alike).
candidates = (df["exclusion_reason"] == "") & ~df["name_missing"]
df["concept_key"] = None
df.loc[candidates, "concept_key"] = (
    df.loc[candidates, "Name"].str.lower()
    + " || "
    + df.loc[candidates, "ingredients"].apply(lambda x: ",".join(sorted(x)))
)
df["concept_dup_extra"] = candidates & df["concept_key"].duplicated(keep="first")
print("Conceptual duplicate rows (extra copies):", df["concept_dup_extra"].sum())

if DROP_CONCEPTUAL_DUPLICATES:
    set_reason(df["concept_dup_extra"], "conceptual_duplicate_extra_copy")

# ---------------------------------------------------------------
# 8. Save the outputs
# ---------------------------------------------------------------
clean = df[df["exclusion_reason"] == ""].copy()
clean["RecipeId"] = clean["RecipeId"].astype("int64")
clean["ingredients_text"] = clean["ingredients"].apply(lambda x: "|".join(x))

# the final clean set must have unique ids
assert clean["RecipeId"].is_unique, "Clean set still has duplicate RecipeIds!"

clean[["RecipeId", "Name", "RecipeCategory", "category_key", "n_ingredients",
       "ingredients_text", "source_row"]].to_csv(f"{OUT_DIR}recipes_clean.csv", index=False)

# transactions file for the association mining team
# (read it back later with: df["ingredients"] = df["ingredients_text"].str.split("|"))
clean[["RecipeId", "category_key", "ingredients_text"]].to_csv(
    f"{OUT_DIR}recipe_transactions.csv", index=False)

# every row that has any problem, so the team can review it
flag_cols = ["source_row", "RecipeId", "Name", "RecipeCategory", "RecipeIngredientParts",
             "id_ok", "id_conflict", "extra_exact_copy", "name_missing", "name_short",
             "category_missing", "ingredients_ok", "n_ingredients", "concept_dup_extra",
             "exclusion_reason"]
has_problem = (
    (df["exclusion_reason"] != "") | df["name_missing"] | df["name_short"]
    | df["category_missing"] | df["concept_dup_extra"]
)
df.loc[has_problem, flag_cols].to_csv(f"{OUT_DIR}recipes_flagged.csv", index=False)

# ---------------------------------------------------------------
# 9. Data quality report
# ---------------------------------------------------------------
report = {
    "rows_read": len(df),
    "rows_in_clean_set": len(clean),
    "rows_excluded": len(df) - len(clean),
    "invalid_ids": int((~df["id_ok"]).sum()),
    "conflicting_duplicate_id_rows": int(df["id_conflict"].sum()),
    "exact_duplicate_id_extra_rows": int(df["extra_exact_copy"].sum()),
    "missing_names": int(df["name_missing"].sum()),
    "short_names": int(df["name_short"].sum()),
    "missing_categories": int(df["category_missing"].sum()),
    "unknown_format_ingredient_lists": int((~df["ingredients_ok"]).sum()),
    "recipes_with_zero_ingredients": int((df["n_ingredients"] == 0).sum()),
    "recipes_with_duplicate_ingredients": int(df["had_ingredient_dups"].sum()),
    "ingredient_names_containing_pipe": int(df["has_pipe"].sum()),
    "conceptual_duplicate_extra_rows": int(df["concept_dup_extra"].sum()),
    "single_ingredient_recipes_in_clean_set": int((clean["n_ingredients"] == 1).sum()),
    "unique_categories_in_clean_set": int(clean["category_key"].nunique()),
}
print("\n--- Data quality report ---")
for name, value in report.items():
    print(f"{name}: {value}")
print("\nExclusion reasons:")
print(df.loc[df["exclusion_reason"] != "", "exclusion_reason"].value_counts())

pd.Series(report).to_csv(f"{OUT_DIR}data_quality_report.csv", header=["value"])

# ---------------------------------------------------------------
# 10. Plots (saved to files)
# ---------------------------------------------------------------
# Top 15 original categories
top_categories = clean["category_key"].value_counts().head(15)
top_categories.plot(kind="bar", figsize=(10, 5), title="Top 15 original categories")
plt.ylabel("Number of recipes")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}top_categories.png")
plt.close()

# Number of ingredients per recipe
clean["n_ingredients"].plot(kind="hist", bins=range(0, 41), figsize=(8, 4),
                            title="Ingredients per recipe")
plt.xlabel("Number of ingredients")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}ingredients_per_recipe.png")
plt.close()

# Top 20 ingredients (counted once per recipe, because each list has no repeats)
top_ingredients = clean["ingredients"].explode().value_counts().head(20)
top_ingredients.plot(kind="bar", figsize=(10, 5), title="Top 20 ingredients")
plt.ylabel("Number of recipes")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}top_ingredients.png")
plt.close()

print("\nDone. Output files were saved with OUT_DIR =", repr(OUT_DIR))
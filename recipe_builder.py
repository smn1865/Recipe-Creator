"""Create new ingredient combinations from the Food.com recipes data.

How it works (all from recipes.csv, no outside AI):
  1. You give a starting ingredient and (optionally) a category word.
  2. It adds "core" ingredients one at a time: each round it picks an ingredient
     with a strong pairing score (lift) against EVERYTHING already chosen.
     Pairings from well-rated recipes count more than from badly rated ones.
  3. It adds "staples" (salt, butter, oil...) that usually come with that core.
  4. It checks the core combination does not already exist as a real recipe.
  5. It finds the most similar real recipes and borrows amounts and steps
     from them as a template.

Run from the repository root (recipes.csv must be there):
    python ingredient_pairs.py   <- not needed first, this script is standalone
    python recipe_builder.py
Loading takes a minute or two; then you can create as many recipes as you like.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CSV_PATH = Path("recipes.csv")
COLUMNS = [
    "RecipeId", "Name", "RecipeCategory", "RecipeIngredientParts",
    "RecipeIngredientQuantities", "AggregatedRating", "ReviewCount", "RecipeInstructions",
]

# --- settings you can tune ---------------------------------------------------
CORE_SIZE = 5            # ingredients chosen by pairing strength (incl. the starting one)
MAX_STAPLES = 4          # extra common ingredients added at the end
STAPLE_MIN_SHARE = 0.30  # staple must appear in >=30% of recipes with the core ingredients
MIN_INGREDIENT = 30      # ignore ingredients in fewer recipes than this (in the chosen category)
MIN_PAIR = 10            # ignore pairs seen in fewer recipes than this
RATING_PRIOR_REVIEWS = 5 # ratings with few reviews are pulled toward the average
CHOICE_POOL = 3          # pick randomly among the top N candidates, so each run differs
MAX_TRIES = 30           # attempts to find a combination that is not already a real recipe
MAX_SHARED_CORE = 3      # NEW = no real recipe shares more than this many core ingredients
UNKNOWN_PAIR_SCORE = -0.5  # penalty when two ingredients never appear together
MIN_CORE_LIFT = 1.1      # core ingredient must pair at least 10% better than chance
MIN_STAPLE_LIFT = 0.9    # staple must not clash with the core (e.g. no pepper in brownies)

_R_ITEM = re.compile(r'"((?:[^"\\]|\\.)*)"|\bNA\b')


def parse_r_vector(raw) -> list:
    """Turn 'c("1", NA, "1/2")' into ['1', None, '1/2'] (order kept)."""
    if not isinstance(raw, str):
        return []
    out = []
    for m in _R_ITEM.finditer(raw):
        out.append(None if m.group(1) is None else m.group(1).replace('\\"', '"').strip())
    return out


def same_ingredient(a: str, b: str) -> bool:
    """Treat 'tomato' / 'tomatoes' / 'cherry tomatoes' as the same thing."""
    a2, b2 = a.rstrip("s"), b.rstrip("s")
    return a2 in b2 or b2 in a2


class RecipeData:
    def __init__(self, csv_path: Path = CSV_PATH):
        print("Loading recipes (this takes a minute)...")
        df = pd.read_csv(csv_path, usecols=COLUMNS)
        df = df.dropna(subset=["RecipeId", "RecipeIngredientParts"]).reset_index(drop=True)

        self.names = df["Name"].fillna("").tolist()
        self.categories = df["RecipeCategory"].fillna("").str.lower().tolist()
        self.parts = [[(p or "").lower() for p in parse_r_vector(r)] for r in df["RecipeIngredientParts"]]
        self.quantities = df["RecipeIngredientQuantities"].tolist()   # parsed only when needed
        self.instructions = df["RecipeInstructions"].tolist()
        self.ratings = df["AggregatedRating"].tolist()
        self.sets = [frozenset(p for p in parts if p) for parts in self.parts]

        # Rating weight: shrink few-review ratings toward the average, scale around 1.0
        rating = df["AggregatedRating"]
        reviews = df["ReviewCount"].fillna(0)
        mean = rating.mean()
        shrunk = (reviews * rating.fillna(mean) + RATING_PRIOR_REVIEWS * mean) / (reviews + RATING_PRIOR_REVIEWS)
        self.weights = (shrunk / mean).tolist()

        # Which recipes contain each ingredient
        self.index: dict[str, list[int]] = defaultdict(list)
        for rid, s in enumerate(self.sets):
            for ing in s:
                self.index[ing].append(rid)
        print(f"Loaded {len(self.sets):,} recipes, {len(self.index):,} distinct ingredients.\n")


class Builder:
    """Scores for one category (or all recipes)."""

    def __init__(self, data: RecipeData, category_word: str = ""):
        self.data = data
        word = category_word.lower().strip()
        self.allowed = {i for i, c in enumerate(data.categories) if word in c} if word else None
        ids = self.allowed if self.allowed is not None else range(len(data.sets))

        self.single: Counter = Counter()
        self.total = 0.0
        for i in ids:
            w = data.weights[i]
            self.total += w
            for ing in data.sets[i]:
                self.single[ing] += w
        self.n_recipes = len(ids)
        self._cooc_cache: dict[str, Counter] = {}

    def recipes_with(self, ing: str) -> list[int]:
        ids = self.data.index.get(ing, [])
        return ids if self.allowed is None else [i for i in ids if i in self.allowed]

    def cooc(self, ing: str) -> Counter:
        """Weighted count of each other ingredient appearing with `ing`."""
        if ing not in self._cooc_cache:
            c: Counter = Counter()
            for i in self.recipes_with(ing):
                w = self.data.weights[i]
                for other in self.data.sets[i]:
                    c[other] += w
            self._cooc_cache[ing] = c
        return self._cooc_cache[ing]

    def lift(self, a: str, b: str) -> float | None:
        together = self.cooc(a).get(b, 0.0)
        if together < MIN_PAIR:
            return None
        return together * self.total / (self.single[a] * self.single[b])

    def candidate_score(self, cand: str, chosen: list[str]) -> float:
        logs = []
        for c in chosen:
            lf = self.lift(c, cand)
            logs.append(math.log(lf) if lf else UNKNOWN_PAIR_SCORE)
        return sum(logs) / len(logs)

    def build_core(self, seed: str, rng: random.Random) -> list[str]:
        chosen = [seed]
        while len(chosen) < CORE_SIZE:
            pool = set()
            for c in chosen:  # candidates: anything seen with a chosen ingredient
                pool.update(k for k, v in self.cooc(c).items() if v >= MIN_PAIR)
            scored = []
            for cand in pool:
                if self.single[cand] < MIN_INGREDIENT or any(same_ingredient(cand, c) for c in chosen):
                    continue
                s = self.candidate_score(cand, chosen)
                if s > math.log(MIN_CORE_LIFT):  # must clearly beat chance on average
                    scored.append((s, cand))
            if not scored:
                break
            scored.sort(reverse=True)
            chosen.append(rng.choice(scored[:CHOICE_POOL])[1])
        return chosen

    def add_staples(self, core: list[str]) -> list[str]:
        """Common ingredients that usually come with the core (salt, butter...)."""
        shares = Counter()
        for c in core:
            base = self.single[c]
            for k, v in self.cooc(c).items():
                shares[k] += (v / base) / len(core)
        staples = []
        for ing, share in shares.most_common():
            if share < STAPLE_MIN_SHARE or len(staples) >= MAX_STAPLES:
                break
            if any(same_ingredient(ing, c) for c in core + staples):
                continue
            if self.candidate_score(ing, core) < math.log(MIN_STAPLE_LIFT):
                continue
            staples.append(ing)
        return staples

    def closest_existing(self, ingredients: list[str]) -> tuple[int, int]:
        """Most core ingredients any real recipe (in ALL data) shares, and how many recipes do."""
        shared = Counter()
        for ing in ingredients:
            for i in self.data.index.get(ing, []):
                shared[i] += 1
        if not shared:
            return 0, 0
        top = max(shared.values())
        return top, sum(1 for v in shared.values() if v == top)

    def most_similar(self, ingredients: list[str], n: int = 3) -> list[tuple[float, int]]:
        target = set(ingredients)
        overlap = Counter()
        for ing in target:
            for i in self.recipes_with(ing):
                overlap[i] += 1
        scored = []
        for i, shared in overlap.items():
            if shared < 2:
                continue
            union = len(target | self.data.sets[i])
            scored.append((shared / union, i))
        scored.sort(reverse=True)
        return scored[:n]


def amounts_from_similar(data: RecipeData, similar_ids: list[int], ingredients: list[str]) -> dict[str, str]:
    """Borrow an amount for each ingredient from the most similar recipe that has it."""
    found = {}
    for rid in similar_ids:
        parts = data.parts[rid]
        qty = parse_r_vector(data.quantities[rid])
        if len(qty) != len(parts):  # some rows in the data are misaligned; skip them
            continue
        for p, q in zip(parts, qty):
            if p in ingredients and p not in found and q:
                found[p] = q
    return found


def create_recipe(data: RecipeData, builder: Builder, seed: str, rng: random.Random) -> None:
    seed = seed.lower().strip()
    if builder.single[seed] < MIN_INGREDIENT:
        close = [i for i in builder.single if seed in i and builder.single[i] >= MIN_INGREDIENT]
        print(f"'{seed}' is too rare here. Try one of: {', '.join(sorted(close, key=lambda i: -builder.single[i])[:8]) or 'another ingredient'}\n")
        return

    best = None  # (most shared, how many recipes share it, core)
    for _ in range(MAX_TRIES):
        attempt = builder.build_core(seed, rng)
        top, count = builder.closest_existing(attempt)
        if best is None or (top, count) < best[:2]:
            best = (top, count, attempt)
        if top <= MAX_SHARED_CORE:
            break
    top, count, core = best
    if len(core) < 3:
        print(f"Could only find {len(core)} strong partner(s) for '{seed}' here. "
              "Try a broader category, or press Enter for all recipes.\n")
        return
    staples = builder.add_staples(core)
    full = core + staples

    similar = builder.most_similar(full)
    similar_ids = [i for _, i in similar]
    amounts = amounts_from_similar(data, similar_ids, full)

    print("=" * 70)
    print("RECIPE IDEA:", " & ".join(x.title() for x in core[:3]))
    print("=" * 70)
    if top <= MAX_SHARED_CORE:
        print(f"NEW: no real recipe has more than {top} of these {len(core)} core ingredients.")
    else:
        print(f"(!) Only a variation: {count} real recipe(s) already have {top} of these "
              f"{len(core)} core ingredients. Try again or another starting ingredient.")

    print("\nCore ingredients (chosen by pairing strength):")
    for i, ing in enumerate(core):
        score = "" if i == 0 else f"   avg pairing score {math.exp(builder.candidate_score(ing, core[:i])):.1f}x chance"
        print(f"  - {amounts.get(ing, '?'):>6}  {ing}{score}")
    if staples:
        print("Staples (usually come with these):")
        for ing in staples:
            print(f"  - {amounts.get(ing, '?'):>6}  {ing}")

    print("\nMost similar real recipes (use as a template for steps):")
    for sim, rid in similar:
        rating = data.ratings[rid]
        rating_txt = f"{rating:.1f} stars" if isinstance(rating, float) and not math.isnan(rating) else "no rating"
        print(f"  * {data.names[rid]}  ({sim:.0%} ingredient overlap, {rating_txt})")
    if similar_ids:
        steps = [s for s in parse_r_vector(data.instructions[similar_ids[0]]) if s]
        print(f"\nTemplate steps from '{data.names[similar_ids[0]]}' (adapt by hand):")
        for n, step in enumerate(steps, 1):
            print(f"  {n}. {step}")
    print("\nAmounts marked '?' were not found in similar recipes.\n")


def main() -> None:
    data = RecipeData()
    rng = random.Random()
    builders: dict[str, Builder] = {}
    print("Type a starting ingredient, e.g. chicken. Type 'quit' to stop.")
    while True:
        seed = input("Starting ingredient: ").strip()
        if seed.lower() in {"quit", "exit", "q", ""}:
            break
        word = input("Category word (e.g. dessert, chicken, soup) or Enter for all: ").strip().lower()
        if word not in builders:
            print("Calculating scores...")
            builders[word] = Builder(data, word)
            if builders[word].n_recipes < 300:
                print(f"Warning: only {builders[word].n_recipes} recipes match '{word}'; results may be weak.")
        create_recipe(data, builders[word], seed, rng)


if __name__ == "__main__":
    main()
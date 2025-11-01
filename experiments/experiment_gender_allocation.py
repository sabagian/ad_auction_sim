from collections import Counter
import random
from typing import Dict, Any, Callable, List, Mapping, Optional

from sim.ad_auction import Bidder, AdSpot, Platform


def _group_impressions_by_group(impressions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in impressions:
        grouped.setdefault(entry["group"], []).append(entry)
    return grouped


def calculate_fairness_metrics(
    impressions: List[Dict[str, Any]],
    positive_bidder_name: str | None = None,
    quality_stem: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Compute statistical parity using simulated impression logs.

    Args:
        impressions: per-impression logs generated during the simulation.
        positive_bidder_name: when provided, treat only impressions won by this
            bidder as positive decisions; otherwise, any served ad counts as
            positive.
    """
    grouped = _group_impressions_by_group(impressions)
    positive_rates: Dict[str, float] = {}

    for group, entries in grouped.items():
        total = len(entries)
        if positive_bidder_name is None:
            positives = sum(1 for entry in entries if entry["winner"] is not None)
        else:
            positives = sum(1 for entry in entries if entry["winner"] == positive_bidder_name)
        positive_rates[group] = positives / total if total else 0.0

    max_gap = max(positive_rates.values()) - min(positive_rates.values()) if positive_rates else 0.0

    print("quality stem:", quality_stem)
    print("positive rates:", positive_rates)
    positive_rates_EQ = {'male':positive_rates['male'], 
                         'female':positive_rates['female'], 
                         'female_adjusted': positive_rates['male'] * quality_stem.get('male', 1.0)/ quality_stem.get('female', 1.0) if quality_stem else None}

    positive_rates_EQ['max_gap'] = max(positive_rates_EQ['female_adjusted'], positive_rates_EQ['female']) - min(positive_rates_EQ['female_adjusted'], positive_rates_EQ['female'])

    
    return {
        "per_group": positive_rates,
        "max_gap": max_gap,
        "positive_bidder": positive_bidder_name,
        "adjusted_rates": positive_rates_EQ,
    }



def calculate_total_utility(impressions: List[Dict[str, Any]]) -> float:
    """Sum payments collected by the platform across impressions."""
    return sum(entry.get("price", 0.0) for entry in impressions if entry.get("winner") is not None)


def simple_valuation(bidder: Bidder, adspot: AdSpot, ctrs=None) -> float:
    # Backwards-compatible: accept optional ctrs (ignored) so this function can be
    # directly passed to the simulator which provides ctrs per bidder.
    return sum(bidder.targeting.get(tag, 0.0) for tag in adspot.tags)


def run_simulations(
    n_impressions: int = 2000,
    methods=None,
    seed: int = 0,
    valuation_fn: Callable = simple_valuation,
    bidder_configs: Mapping[str, Mapping[str, float]] | None = None,
    quality_by_group: Mapping[str, Mapping[str, float]] | None = None,
    positive_bidder_name: str | None = None,
    mechanism: str = "efficiency-max",
    underexposed_group: str = "female",
    tie_break_config: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run simulations for a list of auction methods and collect stats.

    Args:
        n_impressions: number of user impressions to simulate
        methods: list of auction methods to simulate (default: ["first_price", "second_price", "gsp"])
        seed: random seed for reproducibility
        bidder_configs: mapping bidder name -> targeting dict, used to build Bidder objects
        valuation_fn: function to compute bidder's valuation for an ad spot
        quality_by_group: mapping group -> bidder quality scores used during allocation
        positive_bidder_name: bidder name considered a positive decision for
            statistical parity (defaults to any winning bidder)
        tie_break_config: optional dictionary forwarded to AdSpot to control
            tie-breaking probabilities between bidders

    Returns a dictionary mapping method -> stats, where stats contains per-gender counts,
    per-bidder spends, average prices, share metrics, fairness metrics, and total utility.
    """
    def quality_fn(adspot: AdSpot, bidder_list: List[Bidder]) -> List[float]:
        group = adspot.tags[0] if adspot.tags else None
        group_scores = quality_by_group.get(group, {})
        return [group_scores.get(b.name, 1.0) for b in bidder_list]
    
    def quality_fn_bidder_specific(adspot: AdSpot, bidder:Bidder) -> List[float]:
        quality_bidder = {}
        for group in quality_by_group:
            group_scores = quality_by_group.get(group, {})
            quality_bidder[group] = group_scores.get(bidder.name, 1.0)
        return quality_bidder
    
    

    if methods is None:
        methods = ["first_price", "second_price", "gsp"]

    random.seed(seed)

    # Define bidders with gender-specific targeting and valuations
    if bidder_configs is None:
        bidder_configs = {
            "Makeup": {"female": 10.0, "male": 2.0},
            "STEM": {"female": 5.0, "male": 0.2},
        }
    bidders = [Bidder(name, dict(targeting)) for name, targeting in bidder_configs.items()]

    if quality_by_group is None:
        quality_by_group = {
            "female": {"Makeup": 0.5, "STEM": 1.0},
            "male": {"Makeup": 0.1, "STEM": 1.0},
        }

    results = {}

    for method in methods:
        platform = Platform(bidders)
        counts = {"male": Counter(), "female": Counter()}
        total_spend = Counter()
        prices_list = []
        impression_log: List[Dict[str, Any]] = []



        for _ in range(n_impressions):
            gender = random.choice(["male", "female"])  # 50/50 distribution
            spot = AdSpot(1, [gender], mechanism=mechanism, tie_break_config=tie_break_config)
            res = platform.assign(
                [spot],
                method=method,
                valuation_fn=valuation_fn,
                quality_fn=quality_fn,
            )[0]
            winner = res["winners"][0]
            price = res["prices"][0]
            prices_list.append(price)

            slot_positions = res.get("slot_positions", spot.pos)
            primary_slot_weight = slot_positions[0] if slot_positions else 1.0
            impression_log.append(
                {
                    "group": gender,
                    "winner": winner.name if winner else None,
                    "price": price,
                    "qualities": res.get("qualities", {}),
                    "slot_weight": primary_slot_weight,
                }
            )
            if winner is None:
                counts[gender]["none"] += 1
            else:
                counts[gender][winner.name] += 1
                total_spend[winner.name] += price

        # compute shares and summary
        summary = {
            "counts": counts,
            "total_spend": dict(total_spend),
            "avg_price": sum(prices_list) / len(prices_list) if prices_list else 0.0,
            "n_impressions": n_impressions,
        }
        quality_stem = quality_fn_bidder_specific(spot, bidder=Bidder("STEM", {}))
        summary["fairness_metrics"] = calculate_fairness_metrics(
            impression_log, positive_bidder_name=positive_bidder_name, quality_stem=quality_stem
        )
        summary["total_utility"] = calculate_total_utility(impression_log)
        # compute per-gender shares for each bidder
        shares = {"male": {}, "female": {}}
        for gender in ["female", "male"]:
            total = sum(counts[gender].values())
            for bidder_name, val in counts[gender].items():
                shares[gender][bidder_name] = val / total if total > 0 else 0.0

        summary["shares"] = shares
        results[method] = summary

    return results


def print_summary(results: Dict[str, Any]):
    for method, stats in results.items():
        print("\nMethod:", method)
        print(f"Total impressions: {stats['n_impressions']}")
        print(f"Average price per impression: {stats['avg_price']:.3f}")
        print("Total spend by bidder:")
        for b, s in stats["total_spend"].items():
            print(f"  {b}: {s:.2f}")
        for gender in ["female", "male"]:
            total = sum(stats["counts"][gender].values())
            print(f"Impressions for {gender}: {total}")
            for name, cnt in stats["counts"][gender].most_common():
                share = stats["shares"][gender].get(name, 0.0)
                print(f"  {name}: {cnt} ({share:.2%})")
        print("---------------------------")
        print("---------------------------")
        print("---------------------------")
        print("---------------------------")
        print("Fairness metrics:")
        fairness_metrics = stats.get("fairness_metrics")
        if fairness_metrics:
            pos_bidder = fairness_metrics.get("positive_bidder")
            if pos_bidder:
                print(f"Statistical parity (share receiving {pos_bidder}):")
            else:
                print("Statistical parity (positive decision rate):")
            for group, rate in fairness_metrics["per_group"].items():
                print(f"  {group}: {rate:.2%}")
            print(f"  Max gap: {fairness_metrics['max_gap']:.2%}")
        print("---------------------------")
        print("---------------------------")
        opportunity = fairness_metrics.get("adjusted_rates") if fairness_metrics else None
        if opportunity:
            print("Equality of Opportunity (adjusted rates):")
            # print male and female adjusted
            for group, rate in opportunity.items():
                if group == 'female_adjusted':
                    print(f"  {group}: {rate:.2%}")
                if group == 'female':
                    print(f"  {group}: {rate:.2%}")
            # check if EQ is respected 
            if opportunity['max_gap'] <= 0.05:
                print(f"  Max gap: {opportunity['max_gap']:.2%} (EQ respected)")
            else:
                print(f"  Max gap: {opportunity['max_gap']:.2%} (EQ NOT respected)")
        print("---------------------------")
        print("---------------------------")

        if "total_utility" in stats:
            print(f"Total platform utility (revenue): {stats['total_utility']:.2f}")


def try_plot(results, out_prefix: str | None = None):
    """Display share plots inline instead of saving to disk."""
    try:
        import matplotlib.pyplot as plt

        for method, stats in results.items():
            genders = ["female", "male"]
            bidders: List[str] = []
            for g in genders:
                for name in stats["counts"][g].keys():
                    if name not in bidders:
                        bidders.append(name)

            data = {b: [stats["shares"][g].get(b, 0.0) for g in genders] for b in bidders}

            x = range(len(genders))
            width = 0.35

            fig, ax = plt.subplots()
            for i, (b, vals) in enumerate(data.items()):
                ax.bar([p + i * width for p in x], vals, width, label=b)

            ax.set_xticks([p + width * (len(data) - 1) / 2 for p in x])
            ax.set_xticklabels(genders)
            ax.set_ylabel('Share of wins')
            ax.set_title(f'{out_prefix}')
            ax.legend()

            plt.show()
            plt.close(fig)

    except ImportError:
        print("Plotting skipped: matplotlib not available.")
    except Exception as e:
        print("Plotting error:", e)


if __name__ == "__main__":
    results = run_simulations(n_impressions=2000, methods=["first_price", "second_price", "gsp"], seed=1)
    print_summary(results)
    try_plot(results)

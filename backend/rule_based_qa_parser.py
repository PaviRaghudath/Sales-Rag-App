from sentence_transformers import SentenceTransformer
import pandas as pd
import re

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def answer_question(query: str, train_data: pd.DataFrame, collection, top_k: int = 3) -> str:
    q = query.lower()

    # Rule 1: Product family with highest total sales
    if re.search(r"(which|what).*family.*(highest|most).*sales", q):
        agg = train_data.groupby("family")["sales"].sum()
        top_family = agg.idxmax()
        value = agg.max()
        return f" Product family with highest total sales: **{top_family}** with **{value:,.2f}** units."

    # Rule 2: Average sales per store
    if re.search(r"(average|mean).*sales.*store", q):
        agg = train_data.groupby("store_nbr")["sales"].mean()
        result = agg.round(2).to_string()
        return f" Average sales per store:\n{result}"

    # Rule 3: Total sales for specific product family
    match = re.search(r"sales.*for.*family\s+(\w+)", q)
    if match:
        family = match.group(1).upper()
        if family in train_data["family"].unique():
            total = train_data[train_data["family"] == family]["sales"].sum()
            return f" Total sales for product family **{family}**: {total:,.2f} units."
        else:
            return f" Product family '{family}' not found in data."

    # Rule 4: Store with highest total sales
    if re.search(r"(which|what).*store.*(highest|most).*sales", q):
        agg = train_data.groupby("store_nbr")["sales"].sum()
        top_store = agg.idxmax()
        value = agg.max()
        return f" Store with highest total sales: **Store {top_store}** with **{value:,.2f}** units."

    # Rule 5: Total overall sales
    if re.search(r"(total|overall).*sales", q):
        total = train_data["sales"].sum()
        return f" Total sales across all stores and families: **{total:,.2f}** units."

    # Rule 6: Monthly sales trends
    if re.search(r"sales by month", q) and "date" in train_data.columns:
        train_data['month'] = pd.to_datetime(train_data['date']).dt.to_period('M')
        agg = train_data.groupby("month")["sales"].sum()
        return f" Monthly sales:\n{agg.round(2).to_string()}"

    # Rule 7: Top family in specific store
    match = re.search(r"(top|highest).*family.*store\s+(\d+)", q)
    if match:
        store = int(match.group(2))
        subset = train_data[train_data["store_nbr"] == store]
        if not subset.empty:
            agg = subset.groupby("family")["sales"].sum()
            top_family = agg.idxmax()
            value = agg.max()
            return f" In store {store}, top product family is **{top_family}** with **{value:,.2f}** units."
        else:
            return f" Store {store} not found in data."

    # Rule 8: Holiday effect on sales
    if "holiday" in q and "sales" in q:
        if "holiday_type" in train_data.columns:
            avg_holiday = train_data[train_data["holiday_type"] != "Work Day"]["sales"].mean()
            avg_non_holiday = train_data[train_data["holiday_type"] == "Work Day"]["sales"].mean()
            return (f" Average sales during holidays: **{avg_holiday:,.2f}** units\n"
                    f" Average sales during non-holidays: **{avg_non_holiday:,.2f}** units")
        else:
            return " No holiday information available in the dataset."

    # Rule 9: Promotion impact on sales
    if "promotion" in q or "onpromotion" in q:
        if "onpromotion" in train_data.columns:
            promo = train_data[train_data["onpromotion"] == 1]["sales"].mean()
            no_promo = train_data[train_data["onpromotion"] == 0]["sales"].mean()
            return (f"💸 Average sales with promotions: **{promo:,.2f}** units\n"
                    f"📉 Average sales without promotions: **{no_promo:,.2f}** units")
        else:
            return " Promotion data not available."

    # Rule 10: Simple sales prediction for next day (mock, not ML)
    if "predict" in q and "tomorrow" in q:
        if "date" in train_data.columns:
            last_day = pd.to_datetime(train_data["date"]).max()
            recent_sales = train_data[train_data["date"] == str(last_day)]["sales"]
            prediction = recent_sales.mean()
            return f"🔮 Predicted average sales for next day ({last_day + pd.Timedelta(days=1)}): **{prediction:,.2f}** units"
        else:
            return " Date column not found for prediction."

    # --- SEMANTIC FALLBACK VIA CHROMADB ---
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    documents = results["documents"][0]
    scores = results["distances"][0]

    responses = [
        f"- {doc} (Similarity: {1 - score:.2f})"
        for doc, score in zip(documents, scores)
    ]

    return f" I couldn't parse that directly, but here are similar entries for: '{query}'\n" + "\n".join(responses)

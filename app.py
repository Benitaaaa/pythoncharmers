from flask import Flask, render_template, request, send_from_directory
import pickle
# import time
from pyvis.network import Network
import math
from collections import defaultdict
import pycountry
import pycountry_convert as pc
import os


app = Flask(__name__)

# Load preprocessed data (Avoids re-scraping every time)
try:
    with open("pythoncharmers/processed_data.pkl", "rb") as f:
        data = pickle.load(f)
        relationships = data["relationships"]
        country_mentions = data["country_mentions"]
        print("Data loaded successfully.")

except FileNotFoundError:
    print("⚠️ No saved data found. Run data extraction first!")

try:
    # Load processed country and organization data
    with open("processed_countries_and_organizations.pkl", "rb") as f:
        country_org_data = pickle.load(f)
        
        country_relationships = country_org_data.get("country_relationships", [])
        country_mentions = country_org_data.get("country_mentions", {})
        organization_relationships = country_org_data.get("organization_relationships", [])
        organization_mentions = country_org_data.get("organization_mentions", {})

        print("✅ Country and Organization data loaded successfully.")

except FileNotFoundError:
    print("⚠️ No country/organization data found. Run data extraction first!")
    country_relationships, country_mentions = [], {}
    organization_relationships, organization_mentions = [], {}

# ✅ Function to get continent
def get_continent(country_name):
    try:
        country_name = country_name.strip()  # ✅ Remove extra spaces
        country_code = None

        # ✅ Handle country lookup errors
        for country in pycountry.countries:
            if country.name.lower() == country_name.lower():
                country_code = country.alpha_2
                break

        if not country_code:
            return "Unknown"  # ✅ Avoid breaking the filter

        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent_map = {
            "NA": "North America", "SA": "South America", "EU": "Europe",
            "AF": "Africa", "AS": "Asia", "OC": "Oceania"
        }
        return continent_map.get(continent_code, "Unknown")
    except Exception as e:
        print(f"⚠️ Error mapping continent for {country_name}: {e}")
        return "Unknown"


# ✅ Function to generate News Pyvis Graph (NO entity_type)
def generate_filtered_pyvis_graph(country="All", sentiment="All", region="All", top_n=30, min_relationships=2):
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    net.force_atlas_2based(gravity=-30, central_gravity=0.02, spring_length=250, spring_strength=0.1)

    # Step 1: Apply Filters
    filtered_relationships = [
        r for r in relationships
        if (country == "All" or r['source'] == country or r['target'] == country) and
           (sentiment == "All" or r['sentiment'].lower() == sentiment.lower()) and
           (region == "All" or get_continent(r['source']) in region or get_continent(r['target']) in region)

    ]

    # Step 2: Keep only the Top N Most Mentioned Countries
    if top_n and top_n != "All":
        top_countries = sorted(country_mentions.items(), key=lambda x: x[1], reverse=True)[:int(top_n)]
        top_countries = {country for country, _ in top_countries}

        filtered_relationships = [
            r for r in filtered_relationships
            if r['source'] in top_countries and r['target'] in top_countries
        ]

    # Step 3: Track relationships and apply minimum threshold filtering
    strong_relationships = defaultdict(int)
    for r in filtered_relationships:
        strong_relationships[(r['source'], r['target'])] += 1

    filtered_relationships = [
        r for r in filtered_relationships if strong_relationships[(r['source'], r['target'])] >= int(min_relationships)
    ]

    # Step 4: Add Nodes
    added_nodes = set()
    for relation in filtered_relationships:
        added_nodes.add(relation['source'])
        added_nodes.add(relation['target'])

    for country in added_nodes:
        continent = get_continent(country)
        color = {
            "North America": "red", "South America": "green", "Europe": "blue",
            "Africa": "yellow", "Asia": "purple", "Oceania": "orange"
        }.get(continent, "gray")

        size = max(15, min(50, 10 * math.log1p(country_mentions.get(country, 1))))
        net.add_node(country, label=country, color=color, size=size)

    # Step 5: Add Edges
    for relation in filtered_relationships:
        source, target, sentiment = relation["source"], relation["target"], relation["sentiment"]
        edge_color = "green" if relation.get("sentiment", "Neutral") == "POSITIVE" else \
             "red" if relation.get("sentiment", "Neutral") == "NEGATIVE" else "gray"
        net.add_edge(source, target, width=2, color=edge_color, title=relation["sentence"])

    # ✅ Save the graph
    output_file = "pythoncharmers/static/country_network_filtered.html"
    net.show(output_file)
    # time.sleep(0.7)
    return output_file

# ✅ Function to generate Pyvis Graph for Country & Organization (WITH entity_type)
def generate_filtered_pyvis_graph_pdf(entity_type="country", country="All", sentiment="All", region="All", top_n=30, min_relationships=2):
    """Generates a filtered PyVis graph for countries or organizations, supporting region filtering."""
    # ✅ Select the appropriate dataset (Country vs. Organization)
    relationships = country_relationships if entity_type == "country" else organization_relationships
    mentions = country_mentions if entity_type == "country" else organization_mentions

    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    net.force_atlas_2based(gravity=-30, central_gravity=0.02, spring_length=250, spring_strength=0.1)

    # ✅ Step 1: Apply Filters (Country, Sentiment, Region)
    filtered_relationships = [
        r for r in relationships
        if (country == "All" or r["source"] == country or r["target"] == country)  # ✅ Country filter
        and (sentiment == "All" or r["sentiment"].lower() == sentiment.lower())  # ✅ Sentiment filter
        and (region == "All" or (
                entity_type == "country" and (get_continent(r["source"]) == region or get_continent(r["target"]) == region)
        ))  # ✅ Region filter (Only for countries, not organizations)
    ]

    # ✅ Step 2: Keep Only the Top N Most Mentioned Entities (Countries or Organizations)
    if top_n and top_n != "All":
        top_entities = sorted(mentions.items(), key=lambda x: x[1], reverse=True)[:int(top_n)]
        top_entities = {entity for entity, _ in top_entities}
        filtered_relationships = [r for r in filtered_relationships if r["source"] in top_entities and r["target"] in top_entities]

    # ✅ Step 3: Apply Minimum Relationship Threshold
    strong_relationships = defaultdict(int)
    for r in filtered_relationships:
        strong_relationships[(r["source"], r["target"])] += 1

    filtered_relationships = [r for r in filtered_relationships if strong_relationships[(r["source"], r["target"])] >= int(min_relationships)]

    # ✅ Step 4: Add Nodes
    added_nodes = set()
    for relation in filtered_relationships:
        added_nodes.add(relation["source"])
        added_nodes.add(relation["target"])

    for entity in added_nodes:
        if entity_type == "country":
            continent = get_continent(entity)
            color = {
                "North America": "red", "South America": "green", "Europe": "blue",
                "Africa": "yellow", "Asia": "purple", "Oceania": "orange"
            }.get(continent, "gray")  # ✅ Uses continent colors
        else:
            color = "cyan"  # ✅ Organizations use cyan

        size = max(15, min(50, 10 * math.log1p(mentions.get(entity, 1))))
        net.add_node(entity, label=entity, color=color, size=size)

    # ✅ Step 5: Add Edges (Keep Same Edge Filtering)
    for relation in filtered_relationships:
        edge_color = "green" if relation.get("sentiment", "Neutral") == "POSITIVE" else \
                     "red" if relation.get("sentiment", "Neutral") == "NEGATIVE" else "gray"
        net.add_edge(relation["source"], relation["target"], width=2, color=edge_color, title=relation["sentence"])

    # ✅ Save the Graph
    output_file = f"static/{entity_type}_network_filtered_pdf.html"
    net.show(output_file)
    return output_file


# ✅ Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/graph/news")
def graph_news():
    graph_file = generate_filtered_pyvis_graph()
    graph_url = f"/static/{graph_file.split('/')[-1]}"  # Convert file path to URL
    return render_template("graph_wrapper.html", graph_url=graph_url, title="News Relationships")

@app.route("/graph/country")
def graph_country():
    graph_file = generate_filtered_pyvis_graph_pdf(entity_type="country")
    graph_url = f"/static/{graph_file.split('/')[-1]}"  # Convert file path to URL
    return render_template("graph_wrapper.html", graph_url=graph_url, title="Country Relationships")

@app.route("/graph/organization")
def graph_organization():
    graph_file = generate_filtered_pyvis_graph_pdf(entity_type="organization")
    graph_url = f"/static/{graph_file.split('/')[-1]}"  # Convert file path to URL
    return render_template("graph_wrapper.html", graph_url=graph_url, title="Organization Relationships")

@app.route("/update_graph/<graph_type>")
def update_graph(graph_type):
    """Only updates the iframe graph without reloading the full page."""
    country = request.args.get("country", "All")
    sentiment = request.args.get("sentiment", "All")
    region = request.args.get("region", "All")
    top_n = request.args.get("top_n", "30")
    min_relationships = request.args.get("min_relationships", "1")

    if graph_type == "news":
        graph_file = generate_filtered_pyvis_graph(
            country=country, sentiment=sentiment, region=region, top_n=top_n, min_relationships=min_relationships
        )
    else:
        graph_file = generate_filtered_pyvis_graph_pdf(
            entity_type=graph_type, sentiment=sentiment, top_n=top_n, min_relationships=min_relationships
        )

    return f"/static/{os.path.basename(graph_file)}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)

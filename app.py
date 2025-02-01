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
    print("No saved data found. Run data extraction first!")
    relationships, country_mentions = [], {}

# Function to get continent from country name
def get_continent(country_name):
    try:
        country_code = pycountry.countries.lookup(country_name).alpha_2
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent_map = {
            "NA": "North America", "SA": "South America", "EU": "Europe",
            "AF": "Africa", "AS": "Asia", "OC": "Oceania"
        }
        return continent_map.get(continent_code, "Unknown")
    except:
        return "Unknown"

# Function to generate Pyvis graph dynamically based on filters
def generate_filtered_pyvis_graph(country="All", sentiment="All", region="All", top_n=30, min_relationships=2):
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    net.force_atlas_2based(gravity=-30, central_gravity=0.02, spring_length=250, spring_strength=0.1)

    # Step 1: Apply Filters
    filtered_relationships = [
        r for r in relationships
        if (country == "All" or r['source'] == country or r['target'] == country) and
           (sentiment == "All" or r['sentiment'].lower() == sentiment.lower()) and
           (region == "All" or get_continent(r['source']) == region or get_continent(r['target']) == region)
    ]

    # Step 2: Keep only the Top N Most Mentioned Countries (if enabled)
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
        edge_color = "green" if sentiment == "POSITIVE" else "red" if sentiment == "NEGATIVE" else "gray"
        net.add_edge(source, target, width=2, color=edge_color, title=relation["sentence"])

    # Save the graph as an HTML file
    output_file = "pythoncharmers/templates/country_network_filtered.html"
    net.show(output_file)
    # time.sleep(0.7)
    return output_file

# Main Route (Homepage)
@app.route("/")
def home():
    return render_template("index.html")

# Graph Route (Handles Filters & Generates Pyvis Graph)
@app.route("/graph")
def graph():
    country = request.args.get("country", "All")
    sentiment = request.args.get("sentiment", "All")
    region = request.args.get("region", "All")
    top_n = request.args.get("top_n", "30")
    min_relationships = request.args.get("min_relationships", "2")
    print(f"Current Working Directory: {os.getcwd()}")
    graph_file = generate_filtered_pyvis_graph(country, sentiment, region, top_n, min_relationships)
    return send_from_directory("pythoncharmers/templates", "country_network_filtered.html")
    # return render_template("country_network_filtered.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)

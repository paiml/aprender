import yaml

with open("docs/roadmaps/roadmap.yaml") as f:
    data = yaml.safe_load(f)

for item in data.get("roadmap", []):
    if "subtasks" in item and item["subtasks"]:
        new_subtasks = []
        for st in item["subtasks"]:
            if isinstance(st, str):
                new_subtasks.append({"id": st, "title": st, "status": "planned"})
            elif isinstance(st, dict):
                if "id" not in st:
                    st["id"] = st.get("title", "UNKNOWN")
                if "title" not in st:
                    st["title"] = st.get("id", "UNKNOWN")
                if "status" not in st:
                    st["status"] = "planned"
                new_subtasks.append(st)
            else:
                new_subtasks.append(st)
        item["subtasks"] = new_subtasks

with open("docs/roadmaps/roadmap.yaml", "w") as f:
    yaml.dump(data, f, sort_keys=False)

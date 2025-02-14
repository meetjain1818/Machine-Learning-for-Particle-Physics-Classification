import os
import urllib.parse

# GitHub repository details
GITHUB_USER = "meetjain1818"
GITHUB_REPO = "Machine-Learning-for-Particle-Physics-Classification"
GITHUB_BRANCH = "main"  # Change this if using a different branch


notebooks = []
for root, _, files in os.walk("."):  
    for file in files:
        if file.endswith(".ipynb"):
            notebooks.append(os.path.join(root, file))

index_content = "# Machine Learning for Particle Physics Classification - Index\n\n"
index_content += "This repository contains various Jupyter notebooks implementing different ML models.\n\n"
index_content += "## List of Notebooks\n\n"

# Organize notebooks by subfolder
notebooks.sort()
current_folder = ""

for notebook in notebooks:
    relative_path = notebook.replace("\\", "/").lstrip("./")
    folder = os.path.dirname(relative_path)
    filename = os.path.basename(relative_path)
    github_link = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{urllib.parse.quote(relative_path)}"
    # Add folder headers if needed
    if folder and folder != current_folder:
        index_content += f"### 📂 {folder}\n\n"
        current_folder = folder

    index_content += f"- **[{filename}]({github_link})**\n"

with open("INDEX.md", "w", encoding="utf-8") as f:
    f.write(index_content)

print("✅ INDEX.md generated successfully!")

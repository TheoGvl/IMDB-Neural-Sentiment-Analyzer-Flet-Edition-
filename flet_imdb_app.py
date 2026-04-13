import os, re, tarfile, urllib.request, json
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import flet as ft

# --- Basic Settings ---
LINK = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ZIP_FILE = "imdb.tar.gz"
AI_FILE = "model.pth"
DICT_FILE = "words.json"
MAX_WORDS = 10000
MAX_LEN = 200

def fix_text(txt):
    txt = txt.lower().replace("<br />", " ")
    txt = re.sub(r'[^a-z0-9\s]', '', txt)
    return txt

# --- The AI Brain Architecture ---
class Brain(nn.Module):
    def __init__(self, words, dim):
        super().__init__()
        self.emb = nn.Embedding(words, dim)
        self.fc = nn.Linear(dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.emb(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return self.sig(x)

# --- Training Loop that runs only the first time ---
if not os.path.exists(AI_FILE) or not os.path.exists(DICT_FILE):
    print("AI brain not found. Starting training sequence...")
    
    if not os.path.exists(ZIP_FILE):
        print("Downloading dataset (80MB)...")
        urllib.request.urlretrieve(LINK, ZIP_FILE)
    
    texts = []
    scores = []
    
    with tarfile.open(ZIP_FILE, 'r:gz') as tar:
        for m in tar.getmembers():
            if not m.isfile() or not m.name.endswith(".txt"): continue
            if "train/pos/" in m.name:
                f = tar.extractfile(m)
                if f:
                    texts.append(f.read().decode('utf-8'))
                    scores.append(1.0)
            elif "train/neg/" in m.name:
                f = tar.extractfile(m)
                if f:
                    texts.append(f.read().decode('utf-8'))
                    scores.append(0.0)

    counts = Counter()
    clean_texts = []
    for t in texts:
        c = fix_text(t)
        clean_texts.append(c)
        counts.update(c.split())

    top_words = counts.most_common(MAX_WORDS - 1)
    vocab = {w: i + 1 for i, (w, count) in enumerate(top_words)}
    
    with open(DICT_FILE, 'w') as f:
        json.dump(vocab, f)

    class ReviewData(Dataset):
        def __init__(self, data_texts, data_scores):
            self.x = []
            self.y = data_scores
            for t in data_texts:
                ids = [vocab.get(w, 0) for w in t.split()]
                if len(ids) < MAX_LEN: ids += [0] * (MAX_LEN - len(ids))
                else: ids = ids[:MAX_LEN]
                self.x.append(torch.tensor(ids, dtype=torch.long))

        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.x[i], torch.tensor([self.y[i]], dtype=torch.float32)

    data = ReviewData(clean_texts, scores)
    batches = DataLoader(data, batch_size=128, shuffle=True)

    ai = Brain(words=MAX_WORDS, dim=32)
    loss_fn = nn.BCELoss()
    opt = optim.Adam(ai.parameters(), lr=0.005)

    print("⚙️ AI is learning. Please wait...")
    for ep in range(5):
        for x, y in batches:
            opt.zero_grad()
            out = ai(x)
            err = loss_fn(out, y)
            err.backward()
            opt.step()
        print(f"Epoch {ep+1}/5 Completed.")

    torch.save(ai.state_dict(), AI_FILE)
    print("✅ Brain saved successfully!\n")

else:
    print("Ready! Loading saved brain and launching UI...")
    with open(DICT_FILE, 'r') as f:
        vocab = json.load(f)

# --- Load the Model ---
ai = Brain(words=MAX_WORDS, dim=32)
ai.load_state_dict(torch.load(AI_FILE, weights_only=True))
ai.eval()

# --- Flet Graphical Interface ---
def main(page: ft.Page):
    # App window configuration
    page.title = "IMDB Sentiment Engine"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0E0E11"
    page.padding = 40
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    def cast_controls(items: list) -> list[ft.Control]:
        return items

    # UI Elements
    header = ft.Text("Neural Sentiment Analyzer", size=32, weight=ft.FontWeight.W_900, color="#FFFFFF")
    subtitle = ft.Text("Powered by PyTorch & 25,000 IMDB Reviews", size=14, color="#8E8E93")

    input_box = ft.TextField(
        multiline=True,
        min_lines=6,
        max_lines=8,
        hint_text="Paste your movie or TV show review here...",
        bgcolor="#1C1C1E",
        border_color="#2C2C2E",
        width=600,
        text_size=16
    )

    result_text = ft.Text("", size=22, weight=ft.FontWeight.BOLD)

    # Core Action Logic
    def run_analysis(e):
        txt = str(input_box.value).strip()
        if not txt:
            return
            
        c = fix_text(txt)
        ids = [vocab.get(w, 0) for w in c.split()]
        
        if len(ids) < MAX_LEN: ids += [0] * (MAX_LEN - len(ids))
        else: ids = ids[:MAX_LEN]
            
        x = torch.tensor([ids], dtype=torch.long)
        
        with torch.no_grad():
            score = ai(x).item()
            
        if score >= 0.5:
            result_text.value = f"✅ POSITIVE MATCH ({score * 100:.1f}%)"
            result_text.color = "#34C759" # Green
        else:
            result_text.value = f"❌ NEGATIVE MATCH ({(1 - score) * 100:.1f}%)"
            result_text.color = "#FF453A" # Red
            
        page.update()

    def clear_text(e):
        input_box.value = ""
        result_text.value = ""
        page.update()

    # Custom Flet Buttons
    def create_btn(text, icon, color, action):
        return ft.Container(
            content=ft.Row(controls=cast_controls([
                ft.Icon(icon, color="#FFFFFF"), 
                ft.Text(text, color="#FFFFFF", weight=ft.FontWeight.BOLD)
            ])),
            bgcolor=color,
            padding=ft.Padding.symmetric(horizontal=24, vertical=14),
            border_radius=ft.BorderRadius.all(8),
            ink=True,
            on_click=action
        )

    buttons = ft.Row(
        controls=cast_controls([
            create_btn("Clear", ft.Icons.DELETE, "#FF375F", clear_text),
            create_btn("Analyze", ft.Icons.AUTO_AWESOME, "#00D1FF", run_analysis)
        ]),
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=20
    )

    # Build the page layout
    page.add(
        ft.Column(
            controls=cast_controls([
                header,
                subtitle,
                ft.Container(height=30),
                input_box,
                ft.Container(height=15),
                buttons,
                ft.Container(height=30),
                result_text
            ]),
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
    )

ft.run(main)
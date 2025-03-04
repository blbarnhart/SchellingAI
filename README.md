# SchellingAI
# 🏠 Schelling Model with Dynamic Agent Personalities  

This repository contains a **Schelling segregation model** implemented in Python, enhanced with **interactive agent perspectives** powered by **OpenAI’s GPT API**. The simulation allows users to explore how different personality traits affect neighborhood preferences and segregation dynamics.

## 🚀 Features
- 🏠 **Schelling Segregation Model** – Simulates neighborhood segregation based on agent preferences.  
- 🤖 **LLM-Generated Agent Perspectives** – Click on a grid cell to get a **first-person narrative** from an agent.  
- 🎭 **Dynamic Personalities** – Each agent is assigned a **random personality** that affects their tolerance for diverse neighbors.  
- ⚖️ **Adjustable Segregation Thresholds** – Toggle **DYNAMIC_THRESHOLDS** to let personalities influence segregation behavior.  
- 🖱️ **Interactive Visualization** – Click on agents in a Matplotlib-rendered grid to reveal their thoughts.  

## 📌 How It Works
1. The grid initializes with **Red (1) and Blue (2) agents** along with empty spaces (0).  
2. Each agent is assigned a **random personality** from a predefined list (e.g., *“very sociable,”* *“shy and introverted”*).  
3. If **DYNAMIC_THRESHOLDS = True**, an agent’s **segregation tolerance** is determined by their personality.  
4. Clicking on an agent **triggers an LLM-generated response**, giving insight into their happiness or frustration with their neighbors.

## 🛠 Installation & Usage

### 📥 Clone the repository  
```bash
git clone https://github.com/your-username/schelling-agent-perspectives.git
cd schelling-agent-perspectives
```

### 📦 Install dependencies
```bash
pip install numpy matplotlib openai
```

### 🔑 Set your OpenAI API key
```bash
export OPENAI_API_KEY="your-api-key"
```

### ▶️ Run the simulation
```bash
python schelling.py
```

### 🖱️ Click on agents in the grid
After running the script, a Matplotlib window will open. Click on an agent (Red or Blue) to reveal their LLM-generated perspective.

### ⚙️ Customization
You can modify parameters in `schelling.py` to adjust the simulation:

- **Grid size**: Change `GRID_SIZE` to make the simulation larger or smaller.  
- **Empty space ratio**: Adjust `EMPTY_RATIO` to increase or decrease available spots.  
- **Dynamic Thresholds**: Toggle `DYNAMIC_THRESHOLDS = True` to enable **personality-based segregation preferences**.  
- **Custom Personalities**: Modify `PERSONALITY_OPTIONS` and `PERSONALITY_THRESHOLD_MAP` to define **new personality types and tolerance levels**.  
- **LLM Model Choice**: Change `MODEL_NAME` to use `gpt-4` or `gpt-3.5-turbo`.  

---

### 🌱 Future Enhancements
- ⏩ **Step-by-Step Simulation** – Implement movement over time as agents relocate based on their preferences.  
- 🎨 **Graphical Interface** – Replace Matplotlib with a GUI framework for richer interactions.  
- 🏘️ **Neighborhood Effects** – Introduce external factors like economic incentives or peer influence.  

---

### 📜 License
This project is licensed under the **MIT License** – feel free to modify and experiment! 🚀  

---

💡 **Contributions are welcome!** If you have ideas for improvements, feel free to submit a pull request or open an issue.

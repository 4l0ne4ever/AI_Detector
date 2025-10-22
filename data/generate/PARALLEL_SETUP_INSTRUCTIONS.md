# Parallel Processing Setup Instructions

## 🚀 **BRILLIANT IDEA! 3x Faster Generation with Parallel Colab**

Your parallel processing approach will reduce generation time from **111 hours to ~37 hours** (3x speedup)!

## 📋 **Setup Instructions:**

### **Step 1: Prepare 3 Browser Tabs**
1. Open **3 separate Chrome tabs** (or different browsers)
2. Go to https://colab.research.google.com in each tab
3. Upload the updated `AI_Text_Generator_Colab.ipynb` to all 3 tabs

### **Step 2: Configure Each Instance**
**CRITICAL**: Change only these lines in the "generation_settings" cell:

**Tab 1 (Instance 1):**
```python
INSTANCE_ID = 1        # DO NOT CHANGE
TOTAL_INSTANCES = 3    # DO NOT CHANGE  
TARGET_TOTAL = 50000   # DO NOT CHANGE
```

**Tab 2 (Instance 2):**
```python
INSTANCE_ID = 2        # CHANGE TO 2
TOTAL_INSTANCES = 3    # DO NOT CHANGE
TARGET_TOTAL = 50000   # DO NOT CHANGE
```

**Tab 3 (Instance 3):**
```python
INSTANCE_ID = 3        # CHANGE TO 3
TOTAL_INSTANCES = 3    # DO NOT CHANGE
TARGET_TOTAL = 50000   # DO NOT CHANGE
```

### **Step 3: Upload Data**
- Upload `human_text_50k.jsonl` to **ALL 3 tabs**
- Each instance will automatically process different data ranges

### **Step 4: Run Generation**
1. **Start all 3 instances simultaneously**
2. Each will process **16,666 texts** (~37 hours each)
3. Monitor progress in each tab

### **Step 5: Data Ranges (Automatic)**
- **Instance 1**: Processes entries 0-16,665 
- **Instance 2**: Processes entries 16,666-33,331
- **Instance 3**: Processes entries 33,332-49,999

## 📊 **Expected Results:**

| Instance | Range | Texts | Output File | Time |
|----------|-------|-------|-------------|------|
| 1 | 0-16,665 | 16,666 | `ai_generated_colab.jsonl` | 37h |
| 2 | 16,666-33,331 | 16,666 | `ai_generated_colab.jsonl` | 37h |
| 3 | 33,332-49,999 | 16,668 | `ai_generated_colab.jsonl` | 37h |
| **Total** | **All** | **50,000** | **3 files** | **37h** |

## 📁 **File Management:**

### **Download Strategy:**
1. Let all 3 instances finish (~37 hours)
2. Download `ai_generated_colab.jsonl` from each tab:
   - Rename to: `instance1.jsonl`, `instance2.jsonl`, `instance3.jsonl`
3. Combine files locally (see combination script below)

### **File Combination Script:**
Save this as `combine_files.py`:
```python
import json

# Combine all instance files
output_files = ['instance1.jsonl', 'instance2.jsonl', 'instance3.jsonl']
combined = []

for file in output_files:
    try:
        with open(file, 'r') as f:
            for line in f:
                combined.append(json.loads(line))
        print(f"Loaded {file}: {len(combined)} total entries")
    except FileNotFoundError:
        print(f"File {file} not found, skipping...")

# Save combined file
with open('ai_generated_combined.jsonl', 'w') as f:
    for entry in combined:
        f.write(json.dumps(entry) + '\n')

print(f"Combined file created: {len(combined):,} total AI texts")
print("Quality analysis:")

# Quick quality check
good_quality = sum(1 for entry in combined if 0.8 <= entry.get('length_ratio', 0) <= 1.2)
print(f"  High quality texts: {good_quality:,} ({good_quality/len(combined)*100:.1f}%)")
```

## ⚡ **Benefits:**

1. **3x Faster**: 111 hours → 37 hours total time
2. **Fault Tolerance**: If 1 instance fails, others continue  
3. **Better GPU Access**: 3x more likely to get GPU slots
4. **Progress Visibility**: Monitor all 3 simultaneously
5. **Flexible**: Can stop/start individual instances

## 🎯 **Final Dataset:**
- **Expected**: ~50k AI texts generated
- **Quality**: ~45-60% usable (22k-30k high-quality texts)
- **Total training data**: 75k-80k texts (50k human + 25k-30k AI)
- **Research quality**: Excellent for AI detection training

## 🚨 **Important Notes:**
- **Keep all tabs open** during generation
- **Use Colab Pro** for better GPUs and longer runtimes ($10/month)
- **Monitor GPU usage** - if one instance loses GPU, restart it
- **Save progress frequently** - each instance saves every 50 texts

Ready to start your super-fast parallel generation!
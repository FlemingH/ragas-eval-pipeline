import yaml
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, context_recall, context_precision
import os
import pandas as pd
import datetime

"""Load configuration from YAML file"""
# Load config
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config', 'config.yaml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Extract model configurations
embedding_config = config['embedding_model']
evaluate_model_config = config['evaluate_model']

ollama_embeddings_model = OllamaEmbeddings(model=embedding_config['model'])
ollama_evaluate_model = OllamaLLM(model=evaluate_model_config['model'])

data_samples = {
    'question': [
        '张伟是哪个部门的？',
        '张伟是哪个部门的？',
        '张伟是哪个部门的？'
    ],
    'answer': [
        '根据提供的信息，没有提到张伟所在的部门。如果您能提供更多关于张伟的信息，我可能能够帮助您找到答案。',
        '张伟是人事部门的',
        '张伟是教研部的'
    ],
    'ground_truth':[
        '张伟是教研部的成员',
        '张伟是教研部的成员',
        '张伟是教研部的成员'
    ],
    'contexts' : [
        ['提供⾏政管理与协调⽀持，优化⾏政⼯作流程。 ', '绩效管理部 韩杉 李⻜ I902 041 ⼈⼒资源'],
        ['李凯 教研部主任 ', '牛顿发现了万有引力'],
        ['牛顿发现了万有引力', '张伟 教研部工程师，他最近在负责课程研发'],
    ],
}

# Read the Excel file and generate data samples
excel_path = os.path.join(current_dir, 'results', 'test_result_2025525.xlsx')
df = pd.read_excel(excel_path)

# Create data samples from the Excel file
data_samples = {
    'question': df['query'].tolist(),
    'answer': df['TableGPT2 回答'].tolist(),
    'ground_truth': df['golden'].tolist(),

    # Since contexts are not provided in the Excel file, we'll use empty lists with strings
    # You may need to modify this if contexts are available in the Excel file
    'contexts': [[''] for _ in range(len(df))]
}

# Convert any numbers in the data_samples to strings
for key in data_samples:
    if key == 'contexts':
        data_samples[key] = [[str(item) for item in context] for context in data_samples[key]]
    else:
        data_samples[key] = [str(item) for item in data_samples[key]]

dataset = Dataset.from_dict(data_samples)

score = evaluate(
    dataset = dataset,
    metrics=[answer_correctness, context_recall, context_precision],
    llm=ollama_evaluate_model,
    embeddings=ollama_embeddings_model
)

score_df = score.to_pandas()
print(score.to_pandas())

# Save all results to CSV with timestamp and model name
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = evaluate_model_config['model'].replace("/", "_")
csv_dir = os.path.join(current_dir, 'results')
os.makedirs(csv_dir, exist_ok=True)
full_csv_path = os.path.join(csv_dir, f'{model_name}_{timestamp}_results.csv')
score_df.to_csv(full_csv_path, index=False)
print(f"Full results saved to {full_csv_path}")
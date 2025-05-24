import yaml
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness

# Load config
with open('config/api_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract model configurations
embedding_config = config['embedding_model']
evaluate_model_config = config['evaluate_model']

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
    ]
}

dataset = Dataset.from_dict(data_samples)

# Initialize models with config values
llm = ChatOpenAI(
    model=evaluate_model_config['model'],
    openai_api_base=evaluate_model_config['url']
)

embeddings = OpenAIEmbeddings(
    model=embedding_config['model'],
    openai_api_base=embedding_config['url']
)

score = evaluate(
    dataset=dataset,
    metrics=[answer_correctness],
    llm=llm,
    embeddings=embeddings
)
score.to_pandas()
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
!huggingface-cli login

def load_and_preprocess_data(file_path):
    """Load and preprocess the WordPress plugin dataset."""
    df = pd.read_csv(file_path)

    df['last_updated'] = pd.to_datetime(df['last_updated'], format='%Y-%m-%d %I:%M%p GMT', errors='coerce')
    df['added'] = pd.to_datetime(df['added'], format='%Y-%m-%d %I:%M%p GMT', errors='coerce')

    return df



def assess_plugin_quality(df):
    """Perform comprehensive plugin quality assessment."""
    print("Calculating quality scores...")

    df['maintenance_score'] = df.apply(calculate_maintenance_score, axis=1)
    df['popularity_score'] = df.apply(calculate_popularity_score, axis=1)
    df['support_score'] = df.apply(calculate_support_score, axis=1)

    weights = {
        'maintenance_score': 0.4,
        'popularity_score': 0.3,
        'support_score': 0.3
    }

    df['quality_score'] = (
        df['maintenance_score'] * weights['maintenance_score'] +
        df['popularity_score'] * weights['popularity_score'] +
        df['support_score'] * weights['support_score']
    )

    print("Quality assessment completed!")
    return df

def setup_mistral():
    """Setup Mistral model and tokenizer."""
    print("Setting up Mistral model...")

    from transformers import AutoTokenizer, AutoModelForCausalLM


    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model, tokenizer

def analyze_plugin(plugin_data, model, tokenizer):
    """Generate analysis for a single plugin."""
    prompt = f"""<s>[INST] Analyze this WordPress plugin based on its quality metrics:

Plugin: {plugin_data['slug']}

Quality Scores:
- Overall Quality: {plugin_data['quality_score']:.2f}/5.0
- Maintenance: {plugin_data['maintenance_score']:.2f}/5.0
- Popularity: {plugin_data['popularity_score']:.2f}/5.0
- Support: {plugin_data['support_score']:.2f}/5.0

Usage Statistics:
- Active Installations: {plugin_data['active_installs']}
- Total Downloads: {plugin_data['downloaded']}
- Support Thread Resolution: {plugin_data['support_threads_resolved']}/{plugin_data['support_threads']}
- Rating: {plugin_data['rating']}/5.0

Please provide:
1. A brief quality assessment
2. Key strengths and weaknesses
3. Recommendations for improvement [/INST]</s>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=800,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    df = load_and_preprocess_data('wordpress_active_plugins_dataset.csv')

    df_with_scores = assess_plugin_quality(df)

    df_with_scores.to_csv('analyzed_plugins.csv', index=False)

    model, tokenizer = setup_mistral()

    top_5 = df_with_scores.nlargest(5, 'quality_score')

    with open('plugin_analyses_mistral.txt', 'w') as f:
        for _, plugin in top_5.iterrows():
            print(f"\nAnalyzing {plugin['slug']}...")
            analysis = analyze_plugin(plugin, model, tokenizer)

            f.write(f"\n{'='*50}\n")
            f.write(f"Analysis for {plugin['slug']}\n")
            f.write(f"{'='*50}\n")
            f.write(analysis)
            f.write(f"\n{'='*50}\n")

            print(f"Analysis completed for {plugin['slug']}")

    print("\nAnalyses completed! Results saved to:")
    print("1. analyzed_plugins.csv - Complete dataset with quality scores")
    print("2. plugin_analyses.txt - Detailed LLM analysis of top plugins")

if __name__ == "__main__":
    main()

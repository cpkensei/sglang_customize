import asyncio
import sglang as sgl
from sglang.srt.conversation import chat_templates
from sglang.utils import async_stream_and_merge, stream_and_merge

# Configure the model path
MODEL_PATH = "/root/autodl-tmp/model_save/llama3_8b_new/sgl_quant_model_awq"



if __name__ == "__main__":
    # Initialize the SGLang engine with the specified model
    llm = sgl.Engine(model_path=MODEL_PATH)

    # Define prompts for inference
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Set sampling parameters
    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95
    }


    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
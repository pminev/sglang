"""
Usage:
python3 local_example_complete.py
"""

import sglang as sgl


@sgl.function
def few_shot_qa(s, question):
    s += """The following are questions with answers.
Q: What is the capital of France?
A: Paris
Q: What is the capital of Germany?
A: Berlin
Q: What is the capital of Italy?
A: Rome
"""
    s += "Q: " + question + "\n"
    s += "A:" + sgl.gen("answer", stop="\n", temperature=0.95)


def single():
    state = few_shot_qa.run(question="What is the capital of the United States?")
    answer = state["answer"].strip().lower()

    assert "washington" in answer, f"answer: {state['answer']}"

    print(state.text())


def stream():
    state = few_shot_qa.run(
        question="What is the capital of the United States?", stream=True
    )

    for out in state.text_iter("answer"):
        print(out, end="", flush=True)
    print()


def batch():
    states = few_shot_qa.run_batch(
        [
            {"question": "What is the capital of the United States?"},
            {"question": "What is the capital of China?"},
        ]
    )

    for s in states:
        print(s["answer"])


if __name__ == "__main__":
    import time
    from huggingface_hub import hf_hub_download

    # Calculate the start time
    start = time.time()
    # runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")

    # model_path = hf_hub_download(
    #     "bartowski/Llama-3.2-3B-Instruct-GGUF",
    #     filename="Llama-3.2-3B-Instruct-Q4_K_L.gguf",
    # )

    # engine = sgl.Engine(model_path=model_path, random_seed=42, cuda_graph_max_bs=2)

    # runtime = sgl.Runtime(
    #     model_path=model_path,
    #     log_level="debug")

    runtime = sgl.Runtime(
        model_path="meta-llama/Llama-3.2-1B",
        log_level="debug")
    sgl.set_default_backend(runtime)

    # Show the results : this can be altered however you like
    print("It took", time.time() - start, "seconds to setup backend!")

    # Run a single request
    print("\n========== single ==========\n")
    single()
    print("It took", time.time() - start, "seconds to finish single!")

    # Stream output
    print("\n========== stream ==========\n")
    stream()

    print("It took", time.time() - start, "seconds to finish stream!")

    # Run a batch of requests
    # print("\n========== batch ==========\n")
    # batch()

    runtime.shutdown()


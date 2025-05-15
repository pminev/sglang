"""
Usage:
python3 local_example_chat.py
"""

import sglang as sgl


@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))


def single():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])


def stream():
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
        stream=True,
    )

    for out in state.text_iter():
        print(out, end="", flush=True)
    print()


def batch():
    states = multi_turn_question.run_batch(
        [
            {
                "question_1": "What is the capital of the United States?",
                "question_2": "List two local attractions.",
            },
            {
                "question_1": "What is the capital of France?",
                "question_2": "What is the population of this city?",
            },
        ]
    )

    for s in states:
        print(s.messages())


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

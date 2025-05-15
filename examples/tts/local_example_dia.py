"""
Usage:
python3 local_example_dia.py
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

if __name__ == "__main__":
    import time

    # Calculate the start time
    start = time.time()

    runtime = sgl.Runtime(
        model_path="nari-labs/Dia-1.6B",
        log_level="debug")
    sgl.set_default_backend(runtime)

    # Show the results : this can be altered however you like
    print("It took", time.time() - start, "seconds to setup backend!")

    # Run a single request
    print("\n========== single ==========\n")
    single()
    print("It took", time.time() - start, "seconds to finish single!")

    runtime.shutdown()


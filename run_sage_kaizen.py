import time

from sage_kaizen_prompt_lib import (
    SageKaizenLLM,
    TemplateKey,
)

if __name__ == "__main__":
    sk = SageKaizenLLM()

    # start_time = time.time()

    result = sk.generate(
        "What is the earliest known civilizations to study the stars, and how did they use this knowledge for religious purposes?",
        templates=(
            TemplateKey.UNIVERSAL_DEPTH_ANCHOR,
            TemplateKey.STRUCTURED_KNOWLEDGE,
            TemplateKey.ANTI_EARLY_STOP,
        ),
    )

    # end_time = time.time()

    response_text = result["text"]
    chosen_quant = result["chosen_quant"]

    # elapsed_time = end_time - start_time

    # with open("output.txt", "a", encoding="utf-8") as f:
    #     f.write(f"\n\nUsing model [{chosen_quant}] Total Time: {elapsed_time}\n{response_text}\n")

    print("\n===== RESPONSE =====\n")
    print(response_text)
    print("\nChosen quant:", chosen_quant)

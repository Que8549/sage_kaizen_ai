

from sage_kaizen_prompt_lib import (
    SageKaizenLLM,
    TemplateKey,
)

if __name__ == "__main__":
    sk = SageKaizenLLM()

    user_prompt = "What is the earliest known civilizations to study the stars, and how did they use this knowledge for religious purposes?"
    result = sk.generate(
        user_prompt,
        templates=(
            TemplateKey.UNIVERSAL_DEPTH_ANCHOR,
            TemplateKey.STRUCTURED_KNOWLEDGE,
            TemplateKey.ANTI_EARLY_STOP,
        ),
    )


    # response_text = result["text"]
    # chosen_quant = result["chosen_quant"]

    # print("\n===== RESPONSE =====\n")
    # print(response_text)
    # print("\nChosen quant:", chosen_quant)

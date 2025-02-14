char_A = "'A'"
char_B = "'B'"
char_NA = "'NA'"

pairwise_system_prompt = (f"You are a helpful assistant that simply responds whether another AI assistant's Response {char_A} "
                 f"or Response {char_B} is better for a given instruction." 
                 f" If neither response clearly adheres to it more or if the Principle is irrelevant, you respond with {char_NA}.")


def form_pairwise_chat_prompt(x, y_a, y_b, attribute):
    return [
            {"role": "system", "content": pairwise_system_prompt},
            {"role": "user", "content":
                f'## Prompt:\n"{x}"\n\n'
                f'## Response {char_A}:\n"{y_a}"\n\n'
                f'## Response {char_B}:\n"{y_b}"\n\n'
                f'## Answer:\n Which AI Response is better for a given instruction {attribute}?\n'
                f'**Answer only with {char_A} or {char_B}.'
                + (f' Respond with {char_NA} if neither response is clearly better.')
                + '**'
             }
        ]


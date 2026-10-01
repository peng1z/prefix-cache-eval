def length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds):
    return len(prompt_token_ids) if prompt_token_ids is not None else prompt_embeds.shape[0]

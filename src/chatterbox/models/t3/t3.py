    @torch.inference_mode()
    def inference_stream(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor] = None,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 0.95,
    ):
        """Yield speech tokens one-by-one.

        This mirrors :meth:`inference` but instead of returning the full
        sequence at the end, tokens are yielded as soon as they are sampled.
        It is intended for streaming TTS where downstream models can consume
        tokens incrementally.

        Args:
            t3_cond: Conditionals for T3.
            text_tokens: Text tokens including BOS/EOS and duplicated for CFG.
            initial_speech_tokens: Optional starting speech tokens.
            max_new_tokens: Maximum number of speech tokens to emit.
            temperature, cfg_weight, repetition_penalty, min_p, top_p: sampling
                parameters that match those of :meth:`inference`.

        Yields:
            torch.Tensor: A tensor with shape ``(1, 1)`` containing the newly
            sampled speech token.
        """

        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Prepare custom input embeddings and keep track of the number of
        # conditioning tokens so we can correctly index the text tokens in the
        # alignment analyzer.
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        self.compiled = False
        if not self.compiled:
            alignment_stream_analyzer = None
            if self.hp.is_multilingual:
                alignment_stream_analyzer = AlignmentStreamAnalyzer(
                    self.tfmr,
                    None,
                    # Use len_cond so the slice covers the text tokens exactly.
                    text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                    alignment_layer_idx=9,
                    eos_idx=self.hp.stop_speech_token,
                )
            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self.compiled = True

        device = embeds.device
        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token) + self.speech_pos_emb.get_fixed_embedding(0)
        # CFG batch: duplicate BOS embed
        bos_embed = torch.cat([bos_embed, bos_embed])
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        generated_ids = bos_token.clone()

        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        output = self.patched_model(
            inputs_embeds=inputs_embeds,
            past_key_values=None,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = output.past_key_values

        for i in range(max_new_tokens):
            logits_step = output.logits[:, -1, :]

            # CFG combine → (1, V)
            cond = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)

            # Optional alignment constraints (multilingual)
            if self.patched_model.alignment_stream_analyzer is not None:
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                last_token = generated_ids[0, -1].item() if generated_ids.size(1) > 0 else None
                logits = self.patched_model.alignment_stream_analyzer.step(
                    logits, next_token=last_token
                )

            # Sampling processors
            ids_for_proc = generated_ids[:1, ...]  # batch = 1 for processors
            logits = repetition_penalty_processor(ids_for_proc, logits)

            if temperature != 1.0:
                logits = logits / temperature

            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Yield immediately for streaming
            yield next_token

            # Stop on EOS
            if next_token.view(-1) == self.hp.stop_speech_token:
                break

            # Append next token with positional embed; duplicate for CFG
            next_token_embed = self.speech_emb(next_token)
            next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = output.past_key_values

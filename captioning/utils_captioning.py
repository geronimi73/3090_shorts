import torch, gc

smolvlm_model, smolvlm_processor = None, None
qwenvlm_model, qwenvlm_processor = None, None
moondream_model, moondream_tokenizer = None, None

def free_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif is_torch_npu_available():
        torch_npu.npu.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

def unload_models():
    smolvlm_model, smolvlm_processor = None, None
    qwenvlm_model, qwenvlm_processor = None, None
    moondream_model, moondream_tokenizer = None

    free_memory()

def load_smolvlm(device="cuda", dtype=torch.bfloat16):
    global smolvlm_model, smolvlm_processor

    if smolvlm_model is None:
        repo = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        print(f"loading {repo}")
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        smolvlm_processor = AutoProcessor.from_pretrained(repo)
        smolvlm_processor.tokenizer.padding_side="left"     
        smolvlm_model = AutoModelForImageTextToText.from_pretrained(
            repo, 
            torch_dtype=dtype,
            # _attn_implementation="flash_attention_2"
        ).to(device)

    return smolvlm_model, smolvlm_processor

def load_qwenvlm(device="cuda", dtype=torch.bfloat16):
    global qwenvlm_model, qwenvlm_processor

    if qwenvlm_model is None:
        repo = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"loading {repo}")
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        qwenvlm_processor = AutoProcessor.from_pretrained(repo)
        qwenvlm_processor.tokenizer.padding_side="left"     
        qwenvlm_model = AutoModelForImageTextToText.from_pretrained(
            repo, 
            torch_dtype=dtype,
            _attn_implementation="flash_attention_2"
        ).to(device)

    return qwenvlm_model, qwenvlm_processor

def load_moondream(device="cuda", dtype=torch.bfloat16):
    global moondream_model, moondream_tokenizer

    if moondream_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = "vikhyatk/moondream2"
        print(f"loading {repo}")
        moondream_model = AutoModelForCausalLM.from_pretrained(
            repo,
            revision="2025-01-09",
            trust_remote_code=True,
        ).to(device)
        moondream_tokenizer = AutoTokenizer.from_pretrained(repo, revision="2025-01-09")

    return moondream_model, moondream_tokenizer


def caption_moondream(img, prompt):
    model, tokenizer = load_moondream()

    return model.query(img, prompt)["answer"].strip()

def batch_caption_moondream(images, prompts):
    model, tokenizer = load_moondream()

    captions_md = model.batch_answer(
        images=images,
        prompts=prompts,
        tokenizer=tokenizer,
    )
    
    return [c.strip() for c in captions_md]


def caption_qwenvlm(img, prompt):
    from qwen_vl_utils import process_vision_info
    model, processor = load_qwenvlm()
    
    conversation = [ 
        dict(role="user", 
             content=[dict(type="image", image=img), dict(type="text", text=prompt)]
        ) 
    ]

    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(conversation)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)
    
    output_ids = model.generate(**inputs, max_new_tokens=128)
    output_ids = output_ids[:, inputs["input_ids"].size(1):]
    outputs = processor.batch_decode(output_ids, skip_special_tokens=True)
    
    return outputs[0].strip().replace("\"","")

def batch_caption_qwenvlm(images, prompts):
    from qwen_vl_utils import process_vision_info
    model, processor = load_qwenvlm()
    
    conversations = [ 
        [dict(role="user", content=[dict(type="image", image=img),dict(type="text", text=prompt)])]
        for img, prompt in zip(images, prompts)
    ]
    texts = [
        processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in conversations
    ]
    image_inputs, video_inputs = process_vision_info(conversations)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)
    
    output_ids = model.generate(**inputs, max_new_tokens=128).cpu()
    output_ids=[ tok_out[len(tok_in):] for tok_in, tok_out in zip(inputs["input_ids"], output_ids) ]     

    generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)

    return [t.strip() for t in generated_texts]

def caption_smolvlm(img, prompt):
    model, processor = load_smolvlm()
    
    conversation = [ 
        dict(role="user", 
             content=[dict(type="image", image=img), dict(type="text", text=prompt)]
        ) 
    ]
    
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=model.dtype)
    
    output_ids = model.generate(**inputs, max_new_tokens=128)
    output_ids = output_ids[:, inputs["input_ids"].size(1):]
    outputs = processor.batch_decode(output_ids, skip_special_tokens=True)
    
    return outputs[0].strip()

def batch_caption_smolvlm2(images, prompts):
    model, processor = load_smolvlm()
    
    conversations = [ 
        [dict(role="user", content=[dict(type="image", image=img),dict(type="text", text=prompt)])]
        for img, prompt in zip(images, prompts)
    ]
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        padding=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    # with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128).cpu()
    output_ids=[ tok_out[len(tok_in):] for tok_in, tok_out in zip(inputs["input_ids"], output_ids) ]     
    generated_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
    
    return [t.strip() for t in generated_texts]
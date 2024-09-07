from transformer_lens.HookedTransformer import HookedTransformer
from ioi_dataset_creation.dataset import IOIDataset
import torch

def get_pythia_70m(device="cuda") -> HookedTransformer:
    tl_model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped")
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

if __name__ == "__main__":
    num_examples = 100
    device = "cuda:0"
    
    tl_model = get_pythia_70m(device=device)
    ioi_dataset = IOIDataset(
        prompt_type="ABBA",
        N=num_examples*2,
        nb_templates=1,
        seed = 0,
    )
    
    abc_dataset = (
        ioi_dataset.gen_flipped_prompts(("IO", "RAND"), seed=1)
        .gen_flipped_prompts(("S", "RAND"), seed=2)
        .gen_flipped_prompts(("S1", "RAND"), seed=3)
    )
    
    seq_len = ioi_dataset.toks.shape[1]
    assert seq_len == 16, f"Well, I thought ABBA #1 was 16 not {seq_len} tokens long..."
    
    default_data = ioi_dataset.toks.long()[:num_examples*2, : seq_len - 1].to(device)
    patch_data = abc_dataset.toks.long()[:num_examples*2, : seq_len - 1].to(device)
    labels = ioi_dataset.toks.long()[:num_examples*2, seq_len-1]
    wrong_labels = torch.as_tensor(ioi_dataset.s_tokenIDs[:num_examples*2], dtype=torch.long, device=device)
    
    assert torch.equal(labels, torch.as_tensor(ioi_dataset.io_tokenIDs, dtype=torch.long))
    labels = labels.to(device)

    validation_data = default_data[:num_examples, :]
    validation_patch_data = patch_data[:num_examples, :]
    validation_labels = labels[:num_examples]
    validation_wrong_labels = wrong_labels[:num_examples]

    test_data = default_data[num_examples:, :]
    test_patch_data = patch_data[num_examples:, :]
    test_labels = labels[num_examples:]
    test_wrong_labels = wrong_labels[num_examples:]
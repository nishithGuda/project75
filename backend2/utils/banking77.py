from datasets import load_dataset
import random
import json
import os
from collections import defaultdict

# Load the dataset
ds = load_dataset("legacy-datasets/banking77")
# Get the training data
train_data = ds["train"]

#print(ds["train"]["label"])
label_names = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", 
    "automatic_top_up", "balance_not_updated_after_bank_transfer", 
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival", 
    "card_delivery_estimate", "card_linking", "card_not_working", 
    "card_payment_fee_charged", "card_payment_not_recognised", 
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card", 
    "contactless_not_working", "country_support", "declined_card_payment", 
    "declined_cash_withdrawal", "declined_transfer", 
    "direct_debit_payment_not_recognised", "disposable_card_limits", 
    "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support", 
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card", 
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", 
    "order_physical_card", "passcode_forgotten", "pending_card_payment", 
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", 
    "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund", 
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account", 
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge", 
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", 
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice", 
    "transfer_fee_charged", "transfer_into_account", 
    "transfer_not_received_by_recipient", "transfer_timing", 
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds", 
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard", 
    "why_verify_identity", "wrong_amount_of_cash_received", 
    "wrong_exchange_rate_for_cash_withdrawal"
]

# Create a dictionary to organize samples by intent label
samples_by_intent = defaultdict(list)
for sample in train_data:
    samples_by_intent[sample['label']].append(sample)

# Calculate how many samples to take from each intent
# For balanced sampling across 77 intents
samples_per_intent = max(1, 1000 // 77)  # About 13 samples per intent

# Collect balanced samples
balanced_samples = []
for intent_label, samples in samples_by_intent.items():
    # Select random samples from this intent, up to samples_per_intent
    selected = random.sample(samples, min(len(samples), samples_per_intent))
    balanced_samples.extend(selected)

# If we don't have 1000 samples yet, add more randomly
if len(balanced_samples) < 1000:
    remaining_needed = 1000 - len(balanced_samples)
    # Flatten all samples into one list
    all_samples = [s for samples in samples_by_intent.values() for s in samples]
    # Get samples we haven't already selected
    remaining_samples = [s for s in all_samples if s not in balanced_samples]
    # Select randomly from remaining samples
    additional_samples = random.sample(remaining_samples, min(len(remaining_samples), remaining_needed))
    balanced_samples.extend(additional_samples)

# Shuffle the balanced samples
random.shuffle(balanced_samples)

# Trim to exactly 1000 if we have more
balanced_samples = balanced_samples[:1000]

# Convert to desired format
formatted_samples = [
    {
        "text": sample["text"],
        "intent": sample["label"],
        "intent_name": label_names[sample["label"]]
    }
    for sample in balanced_samples
]

# Save to JSON file
output_dir = os.path.join("backend2", "datasets")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "banking77_1000_samples.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(formatted_samples, f, indent=4)

print(f"✅ Generated {len(formatted_samples)} balanced samples from Banking77 dataset")
print(f"Saved to {output_file}")

# Print distribution statistics
intents_count = defaultdict(int)
for sample in formatted_samples:
    intents_count[sample["intent_name"]] += 1

print("\nIntent distribution:")
for intent, count in sorted(intents_count.items(), key=lambda x: x[1], reverse=True):
    print(f"{intent}: {count} samples")
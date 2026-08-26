import json
import os

def generate_rubrics():
    os.makedirs("data/raw", exist_ok=True)
    rubrics = [
        {
            "question_id": "q1",
            "question_text": "Explain why plants appear green.",
            "subject": "biology",
            "criteria": [
                {"criterion_id": "q1_c1", "text": "Correctly attributes the green color to chlorophyll reflecting/not absorbing green light", "max_marks": 2},
                {"criterion_id": "q1_c2", "text": "Mentions that red and blue light are absorbed for photosynthesis", "max_marks": 2}
            ]
        },
        {
            "question_id": "q2",
            "question_text": "Describe the core principle of a binary search algorithm.",
            "subject": "computer_science",
            "criteria": [
                {"criterion_id": "q2_c1", "text": "Requires the input array to be sorted", "max_marks": 1},
                {"criterion_id": "q2_c2", "text": "Repeatedly divides the search interval in half", "max_marks": 2}
            ]
        },
        {
            "question_id": "q3",
            "question_text": "What is the primary cause of Earth's seasons?",
            "subject": "astronomy",
            "criteria": [
                {"criterion_id": "q3_c1", "text": "Attributes seasons to the tilt of the Earth's rotational axis", "max_marks": 2},
                {"criterion_id": "q3_c2", "text": "Mentions the Earth's orbit around the sun", "max_marks": 1}
            ]
        },
        {
            "question_id": "q4",
            "question_text": "Explain the concept of opportunity cost in economics.",
            "subject": "economics",
            "criteria": [
                {"criterion_id": "q4_c1", "text": "Defines it as the value of the next best alternative forgone", "max_marks": 2},
                {"criterion_id": "q4_c2", "text": "Provides a valid example involving a tradeoff", "max_marks": 2}
            ]
        },
        {
            "question_id": "q5",
            "question_text": "How does a vaccine work to protect against viral infections?",
            "subject": "biology",
            "criteria": [
                {"criterion_id": "q5_c1", "text": "Introduces a harmless component or weakened form of the virus", "max_marks": 2},
                {"criterion_id": "q5_c2", "text": "Stimulates the immune system to produce antibodies", "max_marks": 2}
            ]
        },
        {
            "question_id": "q6",
            "question_text": "What is the significance of the Magna Carta?",
            "subject": "history",
            "criteria": [
                {"criterion_id": "q6_c1", "text": "Established the principle that everyone is subject to the law, including the king", "max_marks": 2},
                {"criterion_id": "q6_c2", "text": "Guarantees rights of individuals to justice and a fair trial", "max_marks": 2}
            ]
        },
        {
            "question_id": "q7",
            "question_text": "Explain the difference between kinetic and potential energy.",
            "subject": "physics",
            "criteria": [
                {"criterion_id": "q7_c1", "text": "Defines kinetic energy as energy of motion", "max_marks": 1},
                {"criterion_id": "q7_c2", "text": "Defines potential energy as stored energy based on position or state", "max_marks": 1}
            ]
        },
        {
            "question_id": "q8",
            "question_text": "Describe the water cycle.",
            "subject": "earth_science",
            "criteria": [
                {"criterion_id": "q8_c1", "text": "Includes evaporation and transpiration", "max_marks": 1},
                {"criterion_id": "q8_c2", "text": "Includes condensation and precipitation", "max_marks": 1}
            ]
        }
    ]
    with open("data/raw/rubrics.json", "w") as f:
        json.dump(rubrics, f, indent=2)
    return rubrics

def generate_placeholders():
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/test", exist_ok=True)

    # 1. Base fully-correct answers (X-type) for q1-q8
    train_answers = [
        {
            "answer_id": "q1_train_001", "question_id": "q1",
            "answer_text": "Chlorophyll in plant cells absorbs red and blue light for photosynthesis but reflects green light, which is why plants appear green to us.",
            "human_scores": {"q1_c1": 2, "q1_c2": 2}, "style": "x_type", "placeholder": True
        },
        {
            "answer_id": "q2_train_001", "question_id": "q2",
            "answer_text": "A binary search requires the input array to be sorted. It works by repeatedly dividing the search interval in half to locate the target.",
            "human_scores": {"q2_c1": 1, "q2_c2": 2}, "style": "x_type", "placeholder": True
        },
        {
            "answer_id": "q3_train_001", "question_id": "q3",
            "answer_text": "The seasons are caused by the tilt of the Earth's rotational axis as it orbits around the sun.",
            "human_scores": {"q3_c1": 2, "q3_c2": 1}, "style": "x_type", "placeholder": True
        },
        {
            "answer_id": "q4_train_001", "question_id": "q4",
            "answer_text": "Opportunity cost is the value of the next best alternative that is given up when making a choice. For example, if you spend time studying instead of working, the lost wages are the opportunity cost.",
            "human_scores": {"q4_c1": 2, "q4_c2": 2}, "style": "x_type", "placeholder": True
        },
        {
            "answer_id": "q5_train_001", "question_id": "q5",
            "answer_text": "A vaccine introduces a harmless or weakened form of the virus. This stimulates the immune system to produce antibodies, providing future protection.",
            "human_scores": {"q5_c1": 2, "q5_c2": 2}, "style": "x_type", "placeholder": True
        },
        {
            "answer_id": "q6_train_001", "question_id": "q6",
            "answer_text": "The Magna Carta established that the king is subject to the law, just like everyone else. It also guaranteed the right to a fair trial.",
            "human_scores": {"q6_c1": 2, "q6_c2": 2}, "style": "x_type", "placeholder": True
        },
        {
            "answer_id": "q7_train_001", "question_id": "q7",
            "answer_text": "Kinetic energy is the energy an object has due to its motion. Potential energy is stored energy that depends on the object's position.",
            "human_scores": {"q7_c1": 1, "q7_c2": 1}, "style": "x_type", "placeholder": True
        },
        {
            "answer_id": "q8_train_001", "question_id": "q8",
            "answer_text": "The water cycle involves water evaporating from oceans, forming clouds through condensation, and falling back down as precipitation.",
            "human_scores": {"q8_c1": 0.5, "q8_c2": 1}, "style": "x_type", "placeholder": True
        }
    ]

    for ex in train_answers:
        with open(f"data/train/{ex['answer_id']}.json", "w") as f:
            json.dump(ex, f, indent=2)

    # 2. Test examples (Y-type) covering all 8 variants across different questions
    # Includes varying scores and one "wrong direction" case as requested by the user.
    test_answers = [
        # 1. paraphrase (q1) -> expected ~ same score
        {
            "answer_id": "q1_test_001", "question_id": "q1",
            "derived_from_train_id": "q1_train_001", "variant_type": "paraphrase",
            "answer_text": "The green pigment chlorophyll doesn't absorb green wavelengths; it bounces them back. It does absorb red and blue for photosynthesis.",
            "human_scores": {"q1_c1": 2, "q1_c2": 2}, "placeholder": True
        },
        # 2. scattered_evidence (q2) -> expected ~ same score
        {
            "answer_id": "q2_test_001", "question_id": "q2",
            "derived_from_train_id": "q2_train_001", "variant_type": "scattered_evidence",
            "answer_text": "A binary search requires the input array to be sorted. Algorithms are very important in computer science. It works by repeatedly dividing the search interval in half.",
            "human_scores": {"q2_c1": 1, "q2_c2": 2}, "placeholder": True
        },
        # 3. diffuse_padded (q3) -> expected ~ same score
        {
            "answer_id": "q3_test_001", "question_id": "q3",
            "derived_from_train_id": "q3_train_001", "variant_type": "diffuse_padded",
            "answer_text": "Space is vast and contains many galaxies and solar systems. Our solar system has eight planets. The seasons on our planet are caused by the tilt of the Earth's rotational axis as it orbits the sun. Climate change is also affecting our weather patterns.",
            "human_scores": {"q3_c1": 2, "q3_c2": 1}, "placeholder": True
        },
        # 4. partial_credit_shift (q4) -> degraded score
        {
            "answer_id": "q4_test_001", "question_id": "q4",
            "derived_from_train_id": "q4_train_001", "variant_type": "partial_credit_shift",
            "answer_text": "Opportunity cost is the value of the next best alternative that is given up when making a choice.",
            "human_scores": {"q4_c1": 2, "q4_c2": 0}, "placeholder": True
        },
        # 5. negation_flipped (q5) -> strongly reduced
        {
            "answer_id": "q5_test_001", "question_id": "q5",
            "derived_from_train_id": "q5_train_001", "variant_type": "negation_flipped",
            "answer_text": "A vaccine introduces a harmless form of the virus. This fails to stimulate the immune system to produce antibodies, providing no future protection.",
            "human_scores": {"q5_c1": 2, "q5_c2": 0}, "placeholder": True
        },
        # 6. confidently_wrong (q6) -> strongly reduced
        {
            "answer_id": "q6_test_001", "question_id": "q6",
            "derived_from_train_id": "q6_train_001", "variant_type": "confidently_wrong",
            "answer_text": "The Magna Carta established that the king is above the law and has absolute power over the people. It also guaranteed the right to a fair trial.",
            "human_scores": {"q6_c1": 0, "q6_c2": 2}, "placeholder": True
        },
        # 7. typo_injected (q7) -> expected ~ same score
        {
            "answer_id": "q7_test_001", "question_id": "q7",
            "derived_from_train_id": "q7_train_001", "variant_type": "typo_injected",
            "answer_text": "Kinetic enegy is the energy an object has due to its motion. Potentail energy is stored energy that depends on the object's position.",
            "human_scores": {"q7_c1": 1, "q7_c2": 1}, "placeholder": True
        },
        # 8. genuinely_ambiguous (q8) -> variable/moderate score
        {
            "answer_id": "q8_test_001", "question_id": "q8",
            "derived_from_train_id": "q8_train_001", "variant_type": "genuinely_ambiguous",
            "answer_text": "Water goes up into the sky from the ground and then it comes back down as rain or snow.",
            "human_scores": {"q8_c1": 0.5, "q8_c2": 0.5}, "placeholder": True
        },
        
        # Second round of examples to ensure we have ~16 examples
        
        # 9. paraphrase (q5)
        {
            "answer_id": "q5_test_002", "question_id": "q5",
            "derived_from_train_id": "q5_train_001", "variant_type": "paraphrase",
            "answer_text": "By inserting a weakened variant of the pathogen, vaccinations trigger the body's defenses to generate antibodies.",
            "human_scores": {"q5_c1": 2, "q5_c2": 2}, "placeholder": True
        },
        # 10. scattered_evidence (q4)
        {
            "answer_id": "q4_test_002", "question_id": "q4",
            "derived_from_train_id": "q4_train_001", "variant_type": "scattered_evidence",
            "answer_text": "Opportunity cost is the value of the next best alternative that is given up. Economics is the study of scarcity. For example, if you spend time studying instead of working, the lost wages are the opportunity cost.",
            "human_scores": {"q4_c1": 2, "q4_c2": 2}, "placeholder": True
        },
        # 11. diffuse_padded (q1)
        {
            "answer_id": "q1_test_002", "question_id": "q1",
            "derived_from_train_id": "q1_train_001", "variant_type": "diffuse_padded",
            "answer_text": "Nature is full of amazing colors. The sky is blue and the grass is green. Speaking of grass, chlorophyll in plant cells absorbs red and blue light for photosynthesis but reflects green light, which is why plants appear green to us. Trees also have green leaves.",
            "human_scores": {"q1_c1": 2, "q1_c2": 2}, "placeholder": True
        },
        # 12. partial_credit_shift (q3)
        {
            "answer_id": "q3_test_002", "question_id": "q3",
            "derived_from_train_id": "q3_train_001", "variant_type": "partial_credit_shift",
            "answer_text": "The seasons are caused by the tilt of the Earth's rotational axis.",
            "human_scores": {"q3_c1": 2, "q3_c2": 0}, "placeholder": True
        },
        # 13. negation_flipped (q2)
        {
            "answer_id": "q2_test_002", "question_id": "q2",
            "derived_from_train_id": "q2_train_001", "variant_type": "negation_flipped",
            "answer_text": "A binary search does not require the input array to be sorted. It works by repeatedly dividing the search interval in half.",
            "human_scores": {"q2_c1": 0, "q2_c2": 2}, "placeholder": True
        },
        # 14. confidently_wrong (q7)
        {
            "answer_id": "q7_test_002", "question_id": "q7",
            "derived_from_train_id": "q7_train_001", "variant_type": "confidently_wrong",
            "answer_text": "Kinetic energy is stored energy that depends on the object's position. Potential energy is the energy an object has due to its motion.",
            "human_scores": {"q7_c1": 0, "q7_c2": 0}, "placeholder": True
        },
        # 15. typo_injected (q6)
        {
            "answer_id": "q6_test_002", "question_id": "q6",
            "derived_from_train_id": "q6_train_001", "variant_type": "typo_injected",
            "answer_text": "The Magan Carta established that the king is subject to the law. It also guaranteed the right to a fair trial.",
            "human_scores": {"q6_c1": 2, "q6_c2": 2}, "placeholder": True
        },
        # 16. genuinely_ambiguous (q1)
        {
            "answer_id": "q1_test_003", "question_id": "q1",
            "derived_from_train_id": "q1_train_001", "variant_type": "genuinely_ambiguous",
            "answer_text": "Plants are green because they have some chemicals in their leaves that react with sunlight.",
            "human_scores": {"q1_c1": 0.5, "q1_c2": 0}, "placeholder": True
        },
        # 17. deliberately "wrong direction" case to stress-test pipeline failure detection
        # This is a paraphrase but scored poorly by the "human" to ensure paired-delta catches it as unexpected
        {
            "answer_id": "q8_test_002", "question_id": "q8",
            "derived_from_train_id": "q8_train_001", "variant_type": "paraphrase",
            "answer_text": "The cycle of water consists of evaporation, condensation, and precipitation.",
            "human_scores": {"q8_c1": 0, "q8_c2": 0}, "placeholder": True
        }
    ]

    for ex in test_answers:
        with open(f"data/test/{ex['answer_id']}.json", "w") as f:
            json.dump(ex, f, indent=2)

    # 3. Create Manifest
    manifest = []
    for ex in test_answers:
        manifest.append({
            "test_id": ex["answer_id"],
            "derived_from_train_id": ex["derived_from_train_id"],
            "question_id": ex["question_id"],
            "variant_type": ex["variant_type"]
        })
    with open("data/dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    generate_rubrics()
    generate_placeholders()
    print("Data generation complete.")

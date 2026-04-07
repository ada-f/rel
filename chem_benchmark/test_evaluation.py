"""
Test evaluation functions for chemistry benchmarks using real examples from runs.
"""

import json
from pathlib import Path

from chem_benchmark.evaluation import evaluate_response


def test_c1_examples():
    """Test REL-C1 (isomer set yes/no) evaluation with 3 examples."""
    examples = [
        {
            "question": "Is this list of molecules a set of *constitutional isomers* (same molecular formula, different connectivity)?\n\nSMILES:\n1. CC(C)C(Br)=CBr\n2. CC1(Br)CC1(C)Br\n3. BrC=CCCCBr\n4. C=CCC(C)(Br)Br\n5. CC1CC(Br)C1Br\n\nReturn exactly one of:\n<Yes>\nor\n<No>\nNo explanation.",
            "answer": {"molecules": ["CC(C)C(Br)=CBr", "CC1(Br)CC1(C)Br", "BrC=CCCCBr", "C=CCC(C)(Br)Br", "CC1CC(Br)C1Br"], "label": "Yes"},
            "responses": [
                "<Yes>",  # Correct
                "<No>",   # Incorrect
                "Yes, these are constitutional isomers.",  # Correct (parsed as Yes)
            ]
        },
        {
            "question": "Is this list of molecules a set of *constitutional isomers* (same molecular formula, different connectivity)?\n\nSMILES:\n1. ClC=C=CCCl\n2. ClC1=CCC1Cl\n3. ClC1(Cl)C=CC1\n4. ClC(Cl)C1=CC1\n5. CC(F)C(F)=CF\n\nReturn exactly one of:\n<Yes>\nor\n<No>\nNo explanation.",
            "answer": {"molecules": ["ClC=C=CCCl", "ClC1=CCC1Cl", "ClC1(Cl)C=CC1", "ClC(Cl)C1=CC1", "CC(F)C(F)=CF"], "label": "No"},
            "responses": [
                "<No>",   # Correct
                "<Yes>",  # Incorrect
                "No, they have different molecular formulas.",  # Correct (parsed as No)
            ]
        },
        {
            "question": "Is this list of molecules a set of *constitutional isomers* (same molecular formula, different connectivity)?\n\nSMILES:\n1. BrC1=C2C3C1C23\n2. BrC1C2C3=C2C31\n3. CC12C3=C1C32Br\n4. BrC1=C=C=CC1\n5. BrC=C=C1C=C1\n\nReturn exactly one of:\n<Yes>\nor\n<No>\nNo explanation.",
            "answer": {"molecules": ["BrC1=C2C3C1C23", "BrC1C2C3=C2C31", "CC12C3=C1C32Br", "BrC1=C=C=CC1", "BrC=C=C1C=C1"], "label": "Yes"},
            "responses": [
                "<Yes>",  # Correct
                "<Yes>",  # Correct
                "These molecules are constitutional isomers.",  # Correct (parsed as Yes)
            ]
        },
    ]
    
    print("=" * 80)
    print("Testing REL-C1 (Isomer Set Yes/No) Evaluation")
    print("=" * 80)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Question: {example['question'][:100]}...")
        print(f"Gold Answer: {example['answer']['label']}")
        
        for j, response in enumerate(example['responses'], 1):
            result = evaluate_response(example['question'], example['answer'], response, task="REL-C1")
            print(f"\n  Response {j}: {response[:50]}...")
            print(f"    Predicted: {result.get('pred')}")
            print(f"    Correct: {result.get('correct')}")
            assert result.get('gold') == example['answer']['label'], f"Gold mismatch: {result.get('gold')} != {example['answer']['label']}"


def test_c2_examples():
    """Test REL-C2 (largest common motif) evaluation with 3 examples."""
    examples = [
        {
            "question": "Given the following list of SMILES, what is the largest *connected* common chemical motif (maximum common substructure) present in every molecule?\nRules:\n- The motif must be a single connected fragment.\n- Do NOT tautomerize molecules.\n- Ignore stereochemistry unless it is explicitly encoded and required.\n\nSMILES:\n1. O=C1CCC(C(=O)Oc2ccc(O)cc2)N1\n2. CC(=O)Nc1ccc(OC(=O)C2CCC(=O)N2)cc1\n3. CC1CC(OC(=O)C2CCC(=O)N2)CC(C)(C)C1\n4. O=C1CCC(C(=O)NC2CC2c2ccccc2)N1\n5. O=C1CCC(C(=O)N2CSCC2C(=O)O)N1\n\nReturn your final answer as a single SMILES wrapped exactly like:\n<smiles>YOUR_SMILES_HERE</smiles>\nNo explanation.",
            "answer": {"molecules": ["O=C1CCC(C(=O)Oc2ccc(O)cc2)N1", "CC(=O)Nc1ccc(OC(=O)C2CCC(=O)N2)cc1", "CC1CC(OC(=O)C2CCC(=O)N2)CC(C)(C)C1", "O=C1CCC(C(=O)NC2CC2c2ccccc2)N1", "O=C1CCC(C(=O)N2CSCC2C(=O)O)N1"], "smiles": "O=CC1CCC(=O)N1"},
            "responses": [
                "<smiles>O=C1CCC(C=O)N1</smiles>",  # Correct (same molecule, different SMILES representation)
                "<smiles>O=c1[nH]c(=O)ncc1</smiles>",  # Incorrect (different molecule)
                "<smiles>O=CC1CCC(=O)N1</smiles>",  # Correct (same molecule)
            ]
        },
        {
            "question": "Given the following list of SMILES, what is the largest *connected* common chemical motif (maximum common substructure) present in every molecule?\nRules:\n- The motif must be a single connected fragment.\n- Do NOT tautomerize molecules.\n- Ignore stereochemistry unless it is explicitly encoded and required.\n\nSMILES:\n1. Cc1cn(C2OC(CO)C(O)C2F)c(=O)[nH]c1=O\n2. O=c1[nH]c(=O)n(C2OC(CO)C(O)C2F)cc1I\n3. O=c1[nH]c(=O)n(C2OC(CO)C(O)C2O)cc1F\n4. Cc1cn(C2CC(F)C(CO)O2)c(=O)[nH]c1=O\n5. Cc1cn(C2CC(O)C(CO)O2)c(=O)[nH]c1=O\n\nReturn your final answer as a single SMILES wrapped exactly like:\n<smiles>YOUR_SMILES_HERE</smiles>\nNo explanation.",
            "answer": {"molecules": ["Cc1cn(C2OC(CO)C(O)C2F)c(=O)[nH]c1=O", "O=c1[nH]c(=O)n(C2OC(CO)C(O)C2F)cc1I", "O=c1[nH]c(=O)n(C2OC(CO)C(O)C2O)cc1F", "Cc1cn(C2CC(F)C(CO)O2)c(=O)[nH]c1=O", "Cc1cn(C2CC(O)C(CO)O2)c(=O)[nH]c1=O"], "smiles": "O=c1ccn(C2CCC(CO)O2)c(=O)[nH]1"},
            "responses": [
                "<smiles>O=c1ccn(C2CCC(CO)O2)c(=O)[nH]1</smiles>",  # Correct (same molecule)
                "<smiles>O=c1[nH]c(=O)ncc1</smiles>",  # Incorrect (different molecule, just a substructure)
                "<smiles>c1ccccc1</smiles>",  # Incorrect (different molecule)
            ]
        },
        {
            "question": "Given the following list of SMILES, what is the largest *connected* common chemical motif (maximum common substructure) present in every molecule?\nRules:\n- The motif must be a single connected fragment.\n- Do NOT tautomerize molecules.\n- Ignore stereochemistry unless it is explicitly encoded and required.\n\nSMILES:\n1. CCC(C(C)O)n1ncn(-c2ccc(N3CCN(c4ccc(OCC5COC(Cn6cncn6)(c6ccc(F)cc6F)C5)cc4)CC3)cc2)c1=O\n2. Cc1cc(N2CCN(c3ccc(C(=O)Nc4ccc(F)cc4)cc3)CC2)ccc1OCC1COC(Cn2cncn2)(c2ccc(F)cc2F)C1\n3. CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OCC5COC(CSc6nncn6C)(c6ccc(Cl)cc6)O5)cc4)CC3)cc2)c1=O\n4. CCCCOCCOc1ccc(-c2ccc3c(c2)C=C(C(=O)Nc2ccc([S+]([O-])Cc4cncn4CCC)cc2)CCCN3CC(C)C)cc1\n5. COC(=O)C1=C(C)NC(C)=C(C(=O)OCCc2ccc(N3CCN(C(c4ccccc4)c4ccccc4)CC3)cc2)C1c1cccc([N+](=O)[O-])c1\n\nReturn your final answer as a single SMILES wrapped exactly like:\n<smiles>YOUR_SMILES_HERE</smiles>\nNo explanation.",
            "answer": {"molecules": ["CCC(C(C)O)n1ncn(-c2ccc(N3CCN(c4ccc(OCC5COC(Cn6cncn6)(c6ccc(F)cc6F)C5)cc4)CC3)cc2)c1=O", "Cc1cc(N2CCN(c3ccc(C(=O)Nc4ccc(F)cc4)cc3)CC2)ccc1OCC1COC(Cn2cncn2)(c2ccc(F)cc2F)C1", "CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OCC5COC(CSc6nncn6C)(c6ccc(Cl)cc6)O5)cc4)CC3)cc2)c1=O", "CCCCOCCOc1ccc(-c2ccc3c(c2)C=C(C(=O)Nc2ccc([S+]([O-])Cc4cncn4CCC)cc2)CCCN3CC(C)C)cc1", "COC(=O)C1=C(C)NC(C)=C(C(=O)OCCc2ccc(N3CCN(C(c4ccccc4)c4ccccc4)CC3)cc2)C1c1cccc([N+](=O)[O-])c1"], "smiles": "CCN(CC)c1ccccc1"},
            "responses": [
                "<smiles>c1ccc(N2CCN(c3ccccc3)CC2)cc1</smiles>",  # Incorrect (different molecule, contains correct as substructure)
                "<smiles>CCN(CC)c1ccccc1</smiles>",  # Correct (same molecule)
                "<smiles>c1ccccc1</smiles>",  # Incorrect (different molecule, just a substructure)
            ]
        },
    ]
    
    print("\n" + "=" * 80)
    print("Testing REL-C2 (Largest Common Motif) Evaluation")
    print("=" * 80)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Question: {example['question'][:100]}...")
        print(f"Gold Answer: {example['answer']['smiles']}")
        
        for j, response in enumerate(example['responses'], 1):
            result = evaluate_response(example['question'], example['answer'], response, task="REL-C2")
            print(f"\n  Response {j}: {response[:50]}...")
            print(f"    Predicted: {result.get('pred')}")
            print(f"    Correct: {result.get('correct')}")
            print(f"    Response is substructure of correct: {result.get('response_is_substructure_of_correct')}")
            print(f"    Correct is substructure of response: {result.get('correct_is_substructure_of_response')}")
            print(f"    Overlap metric: {result.get('overlap_metric', 0):.3f}")


def test_c3_examples():
    """Test REL-C3 (missing isomers) evaluation with 3 examples."""
    examples = [
        {
            "question": "Given the following list of constitutional isomers, complete the set by identifying the missing constitutional isomers.\n\nGiven SMILES:\n1. BrCC(Br)C1CC1\n2. CC1CC(Br)C1Br\n3. BrCC1(CBr)CC1\n4. CC1(C)C(Br)C1Br\n5. C=CC(CBr)CBr\n\nReturn the missing molecules as SMILES, one per line, each wrapped exactly like:\n<smiles>YOUR_SMILES_HERE</smiles>\nNo explanation.",
            "answer": {"molecules": ["BrCC(Br)C1CC1", "CC1CC(Br)C1Br", "BrCC1(CBr)CC1", "CC1(C)C(Br)C1Br", "C=CC(CBr)CBr"], "missing_smiles": ["BrC(Br)C1CCC1", "BrC(Br)CC1CC1", "BrC1(Br)CCCC1", "BrC1CCC(Br)C1", "BrC1CCCC1Br"]},
            "responses": [
                "<smiles>BrC(Br)C1CCC1</smiles>\n<smiles>BrC(Br)CC1CC1</smiles>\n<smiles>BrC1(Br)CCCC1</smiles>",  # Partial match
                "<smiles>BrC(Br)C1CCC1</smiles>\n<smiles>BrC(Br)CC1CC1</smiles>\n<smiles>BrC1(Br)CCCC1</smiles>\n<smiles>BrC1CCC(Br)C1</smiles>\n<smiles>BrC1CCCC1Br</smiles>",  # Exact match
                "<smiles>BrC(Br)C1CCC1</smiles>\n<smiles>BrC(Br)CC1CC1</smiles>\n<smiles>BrC1(Br)CCCC1</smiles>\n<smiles>BrC1CCC(Br)C1</smiles>\n<smiles>BrC1CCCC1Br</smiles>\n<smiles>BrC(Br)CCC</smiles>",  # Extra incorrect
            ]
        },
        {
            "question": "Given the following list of constitutional isomers, complete the set by identifying the missing constitutional isomers.\n\nGiven SMILES:\n1. BrC=CC(Br)CBr\n2. CC1(Br)CC1(Br)Br\n3. C=C(Br)CC(Br)Br\n4. BrC(Br)(Br)C1CC1\n5. BrC=CCC(Br)Br\n\nReturn the missing molecules as SMILES, one per line, each wrapped exactly like:\n<smiles>YOUR_SMILES_HERE</smiles>\nNo explanation.",
            "answer": {"molecules": ["BrC=CC(Br)CBr", "CC1(Br)CC1(Br)Br", "C=C(Br)CC(Br)Br", "BrC(Br)(Br)C1CC1", "BrC=CCC(Br)Br"], "missing_smiles": ["BrC(Br)C1(Br)CC1", "BrC(Br)C1CC1Br", "BrC1CC(Br)(Br)C1"]},
            "responses": [
                "<smiles>BrC(Br)C1(Br)CC1</smiles>\n<smiles>BrC(Br)C1CC1Br</smiles>\n<smiles>BrC1CC(Br)(Br)C1</smiles>",  # Exact match
                "<smiles>BrC(Br)C1(Br)CC1</smiles>\n<smiles>BrC(Br)C1CC1Br</smiles>",  # Partial match
                "<smiles>BrC(Br)C1(Br)CC1</smiles>\n<smiles>BrC(Br)C1CC1Br</smiles>\n<smiles>BrC1CC(Br)(Br)C1</smiles>\n<smiles>BrC(Br)CC</smiles>",  # Extra incorrect
            ]
        },
        {
            "question": "Given the following list of constitutional isomers, complete the set by identifying the missing constitutional isomers.\n\nGiven SMILES:\n1. FC1C#CC1\n2. FCC1C#C1\n3. FC1C=C=C1\n4. C=CC#CF\n5. FC1=C2CC12\n\nReturn the missing molecules as SMILES, one per line, each wrapped exactly like:\n<smiles>YOUR_SMILES_HERE</smiles>\nNo explanation.",
            "answer": {"molecules": ["FC1C#CC1", "FCC1C#C1", "FC1C=C=C1", "C=CC#CF", "FC1=C2CC12"], "missing_smiles": ["C#CC(=C)F", "C#CC=CF", "C=C1C=C1F"]},
            "responses": [
                "<smiles>C#CC(=C)F</smiles>\n<smiles>C#CC=CF</smiles>\n<smiles>C=C1C=C1F</smiles>",  # Exact match
                "<smiles>C#CC(=C)F</smiles>\n<smiles>C#CC=CF</smiles>",  # Partial match
                "<smiles>C#CC(=C)F</smiles>\n<smiles>C#CC=CF</smiles>\n<smiles>C=C1C=C1F</smiles>\n<smiles>FC=CC</smiles>",  # Extra incorrect
            ]
        },
    ]
    
    print("\n" + "=" * 80)
    print("Testing REL-C3 (Missing Isomers) Evaluation")
    print("=" * 80)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Question: {example['question'][:100]}...")
        print(f"Gold Answer: {len(example['answer']['missing_smiles'])} missing isomers")
        
        for j, response in enumerate(example['responses'], 1):
            result = evaluate_response(example['question'], example['answer'], response, task="REL-C3")
            print(f"\n  Response {j}: {len(result.get('pred', []))} predicted isomers")
            print(f"    Predicted: {result.get('pred', [])[:3]}..." if len(result.get('pred', [])) > 3 else f"    Predicted: {result.get('pred', [])}")
            print(f"    Correct: {result.get('correct')}")
            print(f"    TP: {result.get('tp')}, FP: {result.get('fp')}, FN: {result.get('fn')}")
            print(f"    Precision: {result.get('precision', 0):.3f}, Recall: {result.get('recall', 0):.3f}, F1: {result.get('f1', 0):.3f}")


def test_c4_examples():
    """Test REL-C4 (constraint satisfaction with motif selection) evaluation with 3 examples."""
    examples = [
        {
            "question": "Given the following 5 molecules, select continuous motifs from exactly 3 molecules such that each motif has at least 6 atoms AND the total number of carboxylic acid groups across all selected motifs equals 2.\n\nMolecules:\n1. CC(C)c1ccc(C(C)C(=O)O)cc1\n2. O=C(O)Cc1ccccc1Cl\n3. CCCCCCc1ccccc1\n4. CC(C)Cc1ccc(O)cc1\n5. O=C(O)c1ccccc1C(=O)O\n\nReturn your answer in this exact format:\n<indices>comma-separated indices of selected molecules (0-indexed)</indices>\n<motif_i>SMILES of motif from molecule i</motif_i>\n\nNo explanation.",
            "answer": {
                "molecules": ["CC(C)c1ccc(C(C)C(=O)O)cc1", "O=C(O)Cc1ccccc1Cl", "CCCCCCc1ccccc1", "CC(C)Cc1ccc(O)cc1", "O=C(O)c1ccccc1C(=O)O"],
                "selected_molecule_indices": [0, 1, 2],
                "selected_motifs": {
                    "0": "C(C)C(=O)O",
                    "1": "O=C(O)Cc1ccccc1",
                    "2": "CCCCCCc1ccccc1"
                }
            },
            "responses": [
                "<indices>0, 1, 2</indices>\n<motif_0>C(C)C(=O)O</motif_0>\n<motif_1>O=C(O)Cc1ccccc1</motif_1>\n<motif_2>CCCCCCc1ccccc1</motif_2>",  # Correct
                "<indices>0, 1, 4</indices>\n<motif_0>C(C)C(=O)O</motif_0>\n<motif_1>O=C(O)Cc1ccccc1</motif_1>\n<motif_4>c1ccccc1</motif_4>",  # Incorrect indices
                "<indices>0, 1, 2</indices>\n<motif_0>CC(C)C</motif_0>\n<motif_1>Cc1ccccc1</motif_1>\n<motif_2>CCCCCC</motif_2>",  # Incorrect motifs
            ]
        },
        {
            "question": "Given the following 5 molecules, select continuous motifs from exactly 3 molecules such that each motif has at least 6 atoms AND the total number of aromatic rings across all selected motifs equals 3.\n\nMolecules:\n1. c1ccc(CCc2ccccc2)cc1\n2. CCCCCCc1ccccc1\n3. CCCCCCCCCC\n4. c1ccc2c(c1)ccc1ccccc12\n5. CC(C)Cc1ccc(O)cc1\n\nReturn your answer in this exact format:\n<indices>comma-separated indices of selected molecules (0-indexed)</indices>\n<motif_i>SMILES of motif from molecule i</motif_i>\n\nNo explanation.",
            "answer": {
                "molecules": ["c1ccc(CCc2ccccc2)cc1", "CCCCCCc1ccccc1", "CCCCCCCCCC", "c1ccc2c(c1)ccc1ccccc12", "CC(C)Cc1ccc(O)cc1"],
                "selected_molecule_indices": [0, 1, 4],
                "selected_motifs": {
                    "0": "c1ccc(CCc2ccccc2)cc1",
                    "1": "CCCCCCc1ccccc1",
                    "4": "CCCCCC"
                }
            },
            "responses": [
                "<indices>0, 1, 4</indices>\n<motif_0>c1ccc(CCc2ccccc2)cc1</motif_0>\n<motif_1>CCCCCCc1ccccc1</motif_1>\n<motif_4>CCCCCC</motif_4>",  # Correct
                "<indices>0, 1, 3</indices>\n<motif_0>c1ccccc1</motif_0>\n<motif_1>c1ccccc1</motif_1>\n<motif_3>c1ccccc1</motif_3>",  # Incorrect (wrong indices)
                "<indices>0, 1, 4</indices>\n<motif_0>CCc1ccccc1</motif_0>\n<motif_1>CCCCCC</motif_1>\n<motif_4>Cc1ccccc1</motif_4>",  # Incorrect (wrong motifs)
            ]
        },
        {
            "question": "Given the following 5 molecules, select continuous motifs from exactly 3 molecules such that each motif has at least 6 atoms AND the total number of primary amine groups across all selected motifs equals 2.\n\nMolecules:\n1. NCCc1ccccc1CCN\n2. CC(C)Cc1ccc(N)cc1\n3. CCCCCCc1ccccc1\n4. O=C(O)Cc1ccccc1Cl\n5. Nc1ccccc1CCCCCN\n\nReturn your answer in this exact format:\n<indices>comma-separated indices of selected molecules (0-indexed)</indices>\n<motif_i>SMILES of motif from molecule i</motif_i>\n\nNo explanation.",
            "answer": {
                "molecules": ["NCCc1ccccc1CCN", "CC(C)Cc1ccc(N)cc1", "CCCCCCc1ccccc1", "O=C(O)Cc1ccccc1Cl", "Nc1ccccc1CCCCCN"],
                "selected_molecule_indices": [0, 1, 2],
                "selected_motifs": {
                    "0": "NCCc1ccccc1",
                    "1": "Cc1ccc(N)cc1",
                    "2": "CCCCCCc1ccccc1"
                }
            },
            "responses": [
                "<indices>0, 1, 2</indices>\n<motif_0>NCCc1ccccc1</motif_0>\n<motif_1>Cc1ccc(N)cc1</motif_1>\n<motif_2>CCCCCCc1ccccc1</motif_2>",  # Correct
                "<indices>0, 4, 2</indices>\n<motif_0>NCCc1ccccc1</motif_0>\n<motif_4>Nc1ccccc1</motif_4>\n<motif_2>CCCCCCc1ccccc1</motif_2>",  # Incorrect (wrong indices)
                "<indices>0, 1, 2</indices>\n<motif_0>c1ccccc1</motif_0>\n<motif_1>CCCCCC</motif_1>\n<motif_2>c1ccccc1</motif_2>",  # Incorrect (wrong motifs, no amines)
            ]
        },
    ]

    print("\n" + "=" * 80)
    print("Testing REL-C4 (Constraint Satisfaction with Motif Selection) Evaluation")
    print("=" * 80)

    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Question: {example['question'][:100]}...")
        print(f"Gold indices: {example['answer']['selected_molecule_indices']}")
        print(f"Gold motifs: {list(example['answer']['selected_motifs'].values())}")

        for j, response in enumerate(example['responses'], 1):
            result = evaluate_response(example['question'], example['answer'], response, task="REL-C4")
            print(f"\n  Response {j}:")
            print(f"    Predicted indices: {result.get('pred_indices')}")
            print(f"    Predicted motifs: {result.get('pred_motifs')}")
            print(f"    Indices correct: {result.get('indices_correct')}")
            print(f"    Motifs correct: {result.get('motifs_correct')}")
            print(f"    Overall correct: {result.get('correct')}")


if __name__ == "__main__":
    test_c1_examples()
    test_c2_examples()
    test_c3_examples()
    test_c4_examples()
    print("\n" + "=" * 80)
    print("All chemistry evaluation tests completed!")
    print("=" * 80)

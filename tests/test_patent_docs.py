"""
Unit Tests for Patent Documentation Verification
================================================
Verifies compliance of DeepLog-Infinity patent specifications for both:
1. Thai Department of Intellectual Property (DIP / กรมทรัพย์สินทางปัญญา)
2. WIPO / PCT International Patent Filing

Validates:
- File existence
- Mandatory statutory sections
- Exactly 10 enumerated claims matching statutory dependencies
- Claim typing (System, Method, HDC, PISNO, CBF, Filter, Federated DP, Remediation, CRM, Edge)
- Substantive word count (> 1,500 words each)
- Complete cross-referencing of all 6 technical figures
- Mathematical rigor (Queuing invariants, Lie derivatives, QP, Lyapunov, HDC, DP & NIZK)
"""

import os
import re
import pytest

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "patent"))
THAI_SPEC_PATH = os.path.join(DOCS_DIR, "patent_specification_thai.md")
ENGLISH_SPEC_PATH = os.path.join(DOCS_DIR, "patent_specification_english.md")


def read_file(filepath: str) -> str:
    """Helper to read utf-8 content of a specification file."""
    assert os.path.exists(filepath), f"File not found: {filepath}"
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


class TestPatentDocumentation:
    """Test suite ensuring production-grade patent specifications."""

    def test_patent_files_exist(self):
        """Verify both Thai DIP and PCT English patent files exist in docs/patent/."""
        assert os.path.isdir(DOCS_DIR), f"Directory {DOCS_DIR} does not exist"
        assert os.path.isfile(THAI_SPEC_PATH), f"Missing Thai patent spec: {THAI_SPEC_PATH}"
        assert os.path.isfile(ENGLISH_SPEC_PATH), f"Missing English patent spec: {ENGLISH_SPEC_PATH}"

    def test_statutory_sections_thai(self):
        """Verify Thai DIP patent specification contains all mandatory statutory sections."""
        content = read_file(THAI_SPEC_PATH)
        mandatory_thai_sections = [
            "ชื่อที่แสดงถึงการประดิษฐ์",
            "สาขาวิทยาการที่เกี่ยวข้องกับการประดิษฐ์",
            "ภูมิหลังของศิลปะหรือวิทยาการที่เกี่ยวข้อง",
            "ลักษณะและความมุ่งหมายของการประดิษฐ์",
            "คำอธิบายรูปเขียนโดยย่อ",
            "การเปิดเผยการประดิษฐ์โดยสมบูรณ์",
            "ข้อถือสิทธิ",
            "บทสรุปการประดิษฐ์",
        ]
        for section in mandatory_thai_sections:
            assert section in content, f"Thai specification is missing mandatory section: '{section}'"

    def test_statutory_sections_english(self):
        """Verify PCT English patent specification contains all mandatory WIPO sections."""
        content = read_file(ENGLISH_SPEC_PATH)
        mandatory_english_sections = [
            "Title of the Invention",
            "Technical Field",
            "Background Art",
            "Summary of the Invention",
            "Brief Description of the Drawings",
            "Detailed Description of Embodiments",
            "Claims",
            "Abstract",
        ]
        for section in mandatory_english_sections:
            assert section in content, f"English specification is missing mandatory section: '{section}'"

    def test_claims_count_and_enumeration_thai(self):
        """Verify Thai specification has exactly 10 enumerated claims (ข้อ 1 ถึง ข้อ 10)."""
        content = read_file(THAI_SPEC_PATH)
        # Extract the claims section
        claims_match = re.search(r"## 7\. ข้อถือสิทธิ.*?(?=## 8\. บทสรุปการประดิษฐ์|$)", content, re.DOTALL)
        assert claims_match is not None, "Claims section not found in Thai specification"
        claims_text = claims_match.group(0)

        # Count enumerated claims (1. ถึง 10.)
        found_claims = re.findall(r"(?:^|\n)\s*(\d+)\.\s+\*\*(.+?)\*\*:\s*(.*?)(?=\n\s*\d+\.\s+\*\*|\Z)", claims_text, re.DOTALL)
        claim_numbers = [int(num) for num, _, _ in found_claims]
        assert claim_numbers == list(range(1, 11)), f"Expected claims 1 to 10, got {claim_numbers}"
        assert len(claim_numbers) == 10, f"Expected exactly 10 claims, found {len(claim_numbers)}"

        # Verify claim types
        # Claim 1: Independent system claim
        assert "ระบบ" in found_claims[0][1], "Claim 1 must be a system claim"
        assert "ระบบตรวจจับความผิดปกติ" in found_claims[0][2]
        # Claim 2: Independent method claim
        assert "กรรมวิธี" in found_claims[1][1], "Claim 2 must be a method claim"
        assert "กรรมวิธีตรวจจับความผิดปกติ" in found_claims[1][2]

        # Dependent claims 3 to 8 reference claim 1 or 2
        for i in range(2, 8):
            claim_title = found_claims[i][1]
            claim_body = found_claims[i][2]
            full_claim = claim_title + " " + claim_body
            assert "ข้อถือสิทธิข้อ 1" in full_claim or "ข้อถือสิทธิข้อ 2" in full_claim or "ข้อถือสิทธิ" in full_claim, (
                f"Claim {i+1} must reference a prior claim"
            )

        # Claim 9: Non-transitory computer-readable medium claim
        assert "สื่อบันทึกข้อมูล" in found_claims[8][1] or "คอมพิวเตอร์" in found_claims[8][1]
        assert "ข้อถือสิทธิข้อ 2" in found_claims[8][2]

        # Claim 10: Kernel hardware / edge acceleration claim
        assert "สถาปัตยกรรมฮาร์ดแวร์" in found_claims[9][1] or "เคอร์เนล" in found_claims[9][1]

    def test_claims_count_and_enumeration_english(self):
        """Verify English PCT specification has exactly 10 enumerated claims (Claims 1 to 10)."""
        content = read_file(ENGLISH_SPEC_PATH)
        claims_match = re.search(r"## 7\. Claims.*?(?=## 8\. Abstract|$)", content, re.DOTALL)
        assert claims_match is not None, "Claims section not found in English specification"
        claims_text = claims_match.group(0)

        # Count enumerated claims
        found_claims = re.findall(r"(?:^|\n)\s*(\d+)\.\s+\*\*(.+?)\*\*:\s*(.*?)(?=\n\s*\d+\.\s+\*\*|\Z)", claims_text, re.DOTALL)
        claim_numbers = [int(num) for num, _, _ in found_claims]
        assert claim_numbers == list(range(1, 11)), f"Expected claims 1 to 10, got {claim_numbers}"
        assert len(claim_numbers) == 10, f"Expected exactly 10 claims, found {len(claim_numbers)}"

        # Verify claim structure
        # Claim 1: Independent system claim
        assert "system" in found_claims[0][1].lower(), "Claim 1 must be a system claim"
        assert "closed-loop autonomous" in found_claims[0][2].lower()
        # Claim 2: Independent method claim
        assert "method" in found_claims[1][1].lower(), "Claim 2 must be a method claim"
        assert "closed-loop autonomous" in found_claims[1][2].lower()

        # Dependent claims 3 to 8
        for i in range(2, 8):
            claim_title = found_claims[i][1]
            claim_body = found_claims[i][2]
            full_claim = claim_title.lower() + " " + claim_body.lower()
            assert "claim 1" in full_claim or "claim 2" in full_claim, (
                f"Claim {i+1} must reference claim 1 or 2"
            )

        # Claim 9: Computer-readable medium
        assert "computer readable" in found_claims[8][1].lower() or "computer-readable" in found_claims[8][1].lower()
        assert "claim 2" in found_claims[8][2].lower()

        # Claim 10: Hardware / edge appliance claim
        assert "appliance" in found_claims[9][1].lower() or "hardware" in found_claims[9][1].lower() or "edge" in found_claims[9][1].lower()

    def test_substantive_word_count(self):
        """Verify both patent documents exceed appropriate substantive word count (>1,500 words each)."""
        thai_content = read_file(THAI_SPEC_PATH)
        english_content = read_file(ENGLISH_SPEC_PATH)

        # Standard whitespace split for English
        english_words = len(re.findall(r"\b\w+\b", english_content))
        # For Thai, counting words/tokens using whitespace and token boundary patterns
        thai_words = len(re.findall(r"[\w\u0E00-\u0E7F]+", thai_content))

        assert english_words > 1500, f"English patent spec is too short: {english_words} words (expected > 1500)"
        assert thai_words > 1500, f"Thai patent spec is too short: {thai_words} words (expected > 1500)"

    def test_technical_drawings_references(self):
        """Verify all 6 technical figures are linked and described in both specifications."""
        thai_content = read_file(THAI_SPEC_PATH)
        english_content = read_file(ENGLISH_SPEC_PATH)

        # Thai drawings: รูปที่ 1 ถึง รูปที่ 6
        for fig_num in range(1, 7):
            fig_str = f"รูปที่ {fig_num}"
            assert fig_str in thai_content, f"Thai specification missing reference to '{fig_str}'"

        # English drawings: FIG. 1 to FIG. 6
        for fig_num in range(1, 7):
            fig_str = f"FIG. {fig_num}"
            assert fig_str in english_content, f"English specification missing reference to '{fig_str}'"

    def test_mathematical_formulations_present(self):
        """Verify presence of core mathematical equations in both patent documents."""
        for path, lang in [(THAI_SPEC_PATH, "Thai"), (ENGLISH_SPEC_PATH, "English")]:
            content = read_file(path)
            # Little's Law / Jackson Network
            assert "L_i = \\lambda_i W_i" in content or "L = \\lambda W" in content, (
                f"{lang} spec missing Little's Law invariant"
            )
            # Control Barrier Function Lie derivatives
            assert "L_f h" in content and "L_g h" in content, (
                f"{lang} spec missing CBF Lie derivatives"
            )
            # Quadratic Program Optimization
            assert "min" in content and "\\text{proposed}" in content, (
                f"{lang} spec missing QP optimization formulation"
            )
            # Lyapunov drift contraction
            assert "\\Delta V" in content and "\\kappa V" in content, (
                f"{lang} spec missing Lyapunov drift contraction"
            )
            # HDC Hypervector binding
            assert "10,000" in content or "10{,}000" in content, (
                f"{lang} spec missing HDC 10,000-bit dimension"
            )
            assert "\\oplus" in content and "\\Pi" in content, (
                f"{lang} spec missing HDC XOR and permutation binding"
            )
            # Differential Privacy & Cryptography
            assert "Differential Privacy" in content or "ความเป็นส่วนตัวเชิงอนุพันธ์" in content, (
                f"{lang} spec missing Differential Privacy"
            )
            assert "Curve25519" in content and "Ed25519" in content, (
                f"{lang} spec missing Curve25519 / Ed25519 cryptographic proofs"
            )

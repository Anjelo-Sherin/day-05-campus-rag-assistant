"""
Pre-loaded Realistic Campus Documents & SQuAD-Style Benchmark Evaluation Suite.

Generates 4 multi-page official campus documents on disk (`.pdf` + `.html`):
1. `CS301_AI_Syllabus_2026.pdf` (3 pages: Grading Breakdown, Late Submission Policy, Attendance & Lab Rules, Exam Schedule)
2. `Academic_Regulations_Handbook.pdf` (3 pages: Minimum Attendance 75%, Medical Leave, CGPA/SGPA Promotion Rules, Plagiarism & AI Integrity Policy, Revaluation)
3. `Campus_Hostel_Library_Circular.pdf` (3 pages: Hostel Gate Curfew Timings, Library Borrowing Limits & Overdue Fines, Campus Wi-Fi & IT Fair Use)
4. `Scholarship_Internship_Guidelines.html` (2 pages: Merit-cum-Means Scholarship Eligibility, Dean's List Stipend, Semester Internship NOC & Credit Transfer)

Also provides a 14-question SQuAD-style ground-truth benchmark dataset (10 in-scope questions with exact file+page targets and 4 out-of-scope adversarial questions) to evaluate:
- Retrieval Hit-Rate@1, Hit-Rate@3, Hit-Rate@K
- Mean Reciprocal Rank (MRR)
- Page-Level Citation Precision
- Answer Faithfulness Score
- Out-of-Scope Hallucination Refusal Accuracy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.document_parser import DocumentPage, build_minimal_pdf, chunk_pages, parse_document
from src.vector_store import CampusVectorStore


CAMPUS_DOCUMENTS_RAW: List[Dict[str, Any]] = [
    {
        "file_name": "CS301_AI_Syllabus_2026.pdf",
        "doc_type": "pdf",
        "pages": [
            (
                "Course Syllabus: CS-301 Artificial Intelligence & Machine Learning (Fall 2026)\n"
                "Instructor: Prof. Ananya Sharma | Office: Turing Block Room 402 | Credits: 4.0\n\n"
                "Section 1: Grading Policy and Evaluation Weightage\n"
                "The final course grade for CS-301 is computed from four mandatory assessment components: "
                "Programming Assignments & Labs carry 25% weightage, the Mid-Semester Examination carries 25% weightage, "
                "the Capstone RAG & Agents Project carries 20% weightage, and the Comprehensive End-Semester Examination "
                "carries 30% weightage. To pass CS-301, a student must score at least 40% overall and at least 35% in "
                "the End-Semester Examination."
            ),
            (
                "Section 2: Late Submission Policy and Slip Days\n"
                "All programming assignments and project milestones are due at 11:59 PM IST on the announced deadline date. "
                "Each student is granted a total of 3 grace slip days for the entire semester, with at most 2 slip days "
                "usable on any single assignment. Once slip days are exhausted, late submissions incur a flat penalty "
                "of 10% per day for up to 3 calendar days. Submissions more than 72 hours (3 days) past the deadline "
                "will not be accepted and receive zero marks.\n\n"
                "Section 3: Weekly Syllabus & Lab Requirements\n"
                "Weeks 1-4 cover Uninformed & Heuristic Search (A*, Minimax) and Probabilistic Graphical Models. "
                "Weeks 5-8 cover Supervised Learning, Neural Networks, and Transformer Architectures. "
                "Weeks 9-12 cover Retrieval-Augmented Generation (RAG), Vector Databases (FAISS/Chroma), "
                "Dense Embeddings (MiniLM/BGE), and Grounded Evaluation Metrics."
            ),
            (
                "Section 4: CS-301 Examination Schedule and Lab Hardware\n"
                "The CS-301 Mid-Semester Exam is scheduled on October 14, 2026 from 09:30 AM to 11:30 AM in Lecture Hall Complex LHC-101. "
                "The End-Semester Exam is scheduled on December 08, 2026 from 02:00 PM to 05:00 PM in LHC-101. "
                "Mid-Semester exams are closed-book, except students may bring one handwritten A4 double-sided formula sheet. "
                "Final Capstone Project demos and code viva will take place on November 28, 2026 in the AI Systems Lab (Turing Block 204)."
            ),
        ],
    },
    {
        "file_name": "Academic_Regulations_Handbook.pdf",
        "doc_type": "pdf",
        "pages": [
            (
                "University Academic Regulations Handbook (2026-2027 Edition)\n\n"
                "Article 4.1: Mandatory Class Attendance Requirement\n"
                "Every enrolled undergraduate and postgraduate student must maintain a minimum attendance of 75% "
                "separately in every registered theory and laboratory course. Students whose attendance falls below 75% "
                "will be debarred (detained) from appearing in the End-Semester Examination for that course and awarded "
                "an 'FA' (Failure due to Attendance) grade.\n\n"
                "Article 4.2: Medical Leave and Attendance Condonation\n"
                "The Dean of Academic Affairs may grant a maximum attendance relaxation of up to 10% (lowering the threshold "
                "from 75% to 65%) strictly for documented hospitalization or representing the university in national competitions. "
                "Medical certificates must be submitted to the Academic Section within 5 working days of returning to campus."
            ),
            (
                "Article 6.1: CGPA Grading Scale, Probation, and Honours\n"
                "The university follows a 10-point Absolute/Relative Letter Grading System: O (Outstanding) = 10 points, "
                "A+ = 9 points, A = 8 points, B+ = 7 points, B = 6 points, C = 5 points, P (Pass) = 4 points, and F (Fail) = 0 points. "
                "A minimum Cumulative Grade Point Average (CGPA) of 5.50 is required for graduation in B.Tech programmes. "
                "Any student whose Semester GPA (SGPA) drops below 5.00 in a regular semester is placed on Academic Probation "
                "and restricted to a maximum credit load of 16 credits in the following semester. "
                "Students graduating with a CGPA of 8.50 or higher and zero historical backlogs are awarded B.Tech with Distinction."
            ),
            (
                "Article 8.3: Academic Integrity, Plagiarism, and Generative AI Policy\n"
                "All submitted theses, lab reports, and code repositories are screened via Turnitin and MOSS similarity checkers. "
                "Text similarity exceeding 15% (excluding bibliography and quoted citations) constitutes academic misconduct. "
                "Use of Generative AI tools (such as Gemini or ChatGPT) is permitted only when explicitly authorized by the "
                "course instructor and must be accompanied by an AI Disclosure Statement detailing prompts and verification steps. "
                "First-time violations result in zero marks for the assessment; second violations result in an 'F' grade in the course.\n\n"
                "Article 9.2: Answer Script Inspection and Revaluation\n"
                "Students may apply for answer script revaluation within 7 calendar days of grade publication by paying a "
                "revaluation fee of Rs. 500 per course. If the revised score increases by 5% or more, the fee is fully refunded."
            ),
        ],
    },
    {
        "file_name": "Campus_Hostel_Library_Circular.pdf",
        "doc_type": "pdf",
        "pages": [
            (
                "Office of the Chief Warden & Central Library - Campus Circular No. 2026/19\n\n"
                "Section 1: Student Hostel Gate Timings and Night Pass Rules\n"
                "All residential hostel main gates close at 11:00 PM on weekdays (Sunday to Thursday) and 11:59 PM on "
                "weekends (Friday and Saturday). Biometric attendance is recorded at the hostel reception between "
                "10:30 PM and 11:15 PM daily. Students accessing the 24x7 Central Reading Room or AI Compute Lab "
                "after 11:00 PM must generate a Digital Night Pass on the Campus Portal approved by their Faculty Advisor. "
                "Day scholars and external visitors are permitted in hostel common rooms only between 09:00 AM and 08:00 PM."
            ),
            (
                "Section 2: Central Library Borrowing Privileges and Overdue Fines\n"
                "Undergraduate students may borrow up to 5 books simultaneously for a loan period of 14 days. "
                "Postgraduate and M.Tech students may borrow up to 8 books for 21 days, and Ph.D. scholars may borrow "
                "up to 12 books for 30 days. A borrowed book may be renewed online once if no reservation hold exists. "
                "Overdue books attract a late fine of Rs. 5 per day per book for the first 7 days, and Rs. 15 per day "
                "thereafter. Reference textbooks and IEEE/ACM bound journals are strictly for in-library consultation."
            ),
            (
                "Section 3: Campus Wi-Fi (EduRoam), GPU Cluster, and Mess Rebate Policy\n"
                "Every student receives a high-speed EduRoam Wi-Fi quota of 50 GB per week on the 5 GHz campus network. "
                "Peer-to-peer torrenting and unauthorized cryptocurrency mining on campus networks or the DGX GPU cluster "
                "lead to immediate MAC address suspension. Students taking approved leave from campus for 5 or more "
                "consecutive days can apply for a hostel mess rebate of Rs. 160 per day via the Student ERP at least "
                "48 hours prior to departure."
            ),
        ],
    },
    {
        "file_name": "Scholarship_Internship_Guidelines.html",
        "doc_type": "webpage",
        "pages": [
            (
                "<h1>Financial Aid, Scholarships & Industrial Internship Guidelines (2026)</h1>\n"
                "<h2>Section 1: Merit-cum-Means (MCM) and Dean's Merit Scholarship</h2>\n"
                "<p>The Merit-cum-Means (MCM) Scholarship provides a 50% tuition fee waiver plus a monthly stipend of "
                "Rs. 3,000 for ten months each academic year. To qualify for the MCM Scholarship, a student must have "
                "an annual gross family income below Rs. 6,00,000 (Six Lakhs), maintain a minimum CGPA of 7.50, and have "
                "no active disciplinary or academic backlog records. Applications close on September 30 every year.</p>\n"
                "<p>The Dean's Merit Scholarship is awarded automatically to the top 5% of students in each branch based "
                "on yearly CGPA (minimum 9.00 CGPA required) and provides a one-time cash award of Rs. 40,000.</p>"
            ),
            (
                "<h2>Section 2: Semester-Long Industrial Internship & NOC Rules</h2>\n"
                "<p>Final-year (7th or 8th semester) B.Tech students with a CGPA of 7.00 or above and zero pending backlogs "
                "are eligible to pursue a 6-month (24-week) external industrial or research internship equivalent to "
                "12 academic credits. Students must submit an official offer letter to the Career Development Cell (CDC) "
                "at least 15 days before the internship start date to obtain a No Objection Certificate (NOC).</p>\n"
                "<p>Evaluation of the 6-month internship requires a Mid-Term Progress Review at Week 10 (30% weight) "
                "and a Final Technical Report + Joint Industry-Faculty Viva at Week 24 (70% weight).</p>"
            ),
        ],
    },
]


EVALUATION_BENCHMARK_SET: List[Dict[str, Any]] = [
    {
        "id": "Q01",
        "question": "What is the minimum attendance percentage required in every course and what is the maximum medical relaxation?",
        "expected_file": "Academic_Regulations_Handbook.pdf",
        "expected_page": 1,
        "expected_keywords": ["75%", "10%", "65%", "debarred"],
        "is_answerable": True,
        "category": "Academic Regulations",
    },
    {
        "id": "Q02",
        "question": "How is the final grade weighted in CS-301 Artificial Intelligence & Machine Learning?",
        "expected_file": "CS301_AI_Syllabus_2026.pdf",
        "expected_page": 1,
        "expected_keywords": ["25%", "20%", "30%", "assignments", "mid-semester", "end-semester"],
        "is_answerable": True,
        "category": "Syllabus & Grading",
    },
    {
        "id": "Q03",
        "question": "What is the late submission penalty and how many grace slip days do students get in CS-301?",
        "expected_file": "CS301_AI_Syllabus_2026.pdf",
        "expected_page": 2,
        "expected_keywords": ["3", "slip", "10%", "72"],
        "is_answerable": True,
        "category": "Syllabus & Grading",
    },
    {
        "id": "Q04",
        "question": "When and where is the CS-301 Mid-Semester exam conducted, and are notes allowed?",
        "expected_file": "CS301_AI_Syllabus_2026.pdf",
        "expected_page": 3,
        "expected_keywords": ["october 14", "lhc-101", "a4", "formula"],
        "is_answerable": True,
        "category": "Exam Schedule",
    },
    {
        "id": "Q05",
        "question": "What CGPA is required for graduation and what happens if a student's SGPA falls below 5.00?",
        "expected_file": "Academic_Regulations_Handbook.pdf",
        "expected_page": 2,
        "expected_keywords": ["5.50", "probation", "16", "credits"],
        "is_answerable": True,
        "category": "Academic Regulations",
    },
    {
        "id": "Q06",
        "question": "What is the Turnitin similarity threshold for plagiarism and the rule on using Generative AI tools?",
        "expected_file": "Academic_Regulations_Handbook.pdf",
        "expected_page": 3,
        "expected_keywords": ["15%", "disclosure", "turnitin", "zero"],
        "is_answerable": True,
        "category": "Academic Integrity",
    },
    {
        "id": "Q07",
        "question": "What are the hostel gate curfew timings on weekdays and weekends?",
        "expected_file": "Campus_Hostel_Library_Circular.pdf",
        "expected_page": 1,
        "expected_keywords": ["11:00", "11:59", "night pass"],
        "is_answerable": True,
        "category": "Campus Circulars",
    },
    {
        "id": "Q08",
        "question": "How many books can an undergraduate borrow from the Central Library and what is the overdue fine per day?",
        "expected_file": "Campus_Hostel_Library_Circular.pdf",
        "expected_page": 2,
        "expected_keywords": ["5", "14", "rs. 5", "rs. 15"],
        "is_answerable": True,
        "category": "Library Rules",
    },
    {
        "id": "Q09",
        "question": "What are the eligibility criteria and stipend amount for the Merit-cum-Means (MCM) Scholarship?",
        "expected_file": "Scholarship_Internship_Guidelines.html",
        "expected_page": 1,
        "expected_keywords": ["50%", "3,000", "6,00,000", "7.50"],
        "is_answerable": True,
        "category": "Scholarships",
    },
    {
        "id": "Q10",
        "question": "What CGPA and credit rules apply for final-year B.Tech students applying for a 6-month industrial internship NOC?",
        "expected_file": "Scholarship_Internship_Guidelines.html",
        "expected_page": 2,
        "expected_keywords": ["7.00", "12", "15", "noc"],
        "is_answerable": True,
        "category": "Internships",
    },
    {
        "id": "Q11",
        "question": "What is the recipe for baking a classic sourdough pizza crust in a wood-fired oven?",
        "expected_file": None,
        "expected_page": None,
        "expected_keywords": [],
        "is_answerable": False,
        "category": "Out-of-Scope (Adversarial)",
    },
    {
        "id": "Q12",
        "question": "Who won the FIFA World Cup penalty shootout in 2022 and how many goals did Mbappe score?",
        "expected_file": None,
        "expected_page": None,
        "expected_keywords": [],
        "is_answerable": False,
        "category": "Out-of-Scope (Adversarial)",
    },
    {
        "id": "Q13",
        "question": "What is the current stock price of NVIDIA and its quarterly GPU revenue guidance?",
        "expected_file": None,
        "expected_page": None,
        "expected_keywords": [],
        "is_answerable": False,
        "category": "Out-of-Scope (Adversarial)",
    },
    {
        "id": "Q14",
        "question": "How do I apply for a Schengen tourist visa at the French consulate in Mumbai?",
        "expected_file": None,
        "expected_page": None,
        "expected_keywords": [],
        "is_answerable": False,
        "category": "Out-of-Scope (Adversarial)",
    },
]


def ensure_sample_campus_docs(sample_dir: Path) -> List[DocumentPage]:
    """
    Ensure the 4 sample campus documents exist on disk in `data/sample_docs/`
    (building real multi-page PDFs and HTML files) and return their parsed `DocumentPage` list.
    """
    sample_dir.mkdir(parents=True, exist_ok=True)
    all_pages: List[DocumentPage] = []

    for doc_spec in CAMPUS_DOCUMENTS_RAW:
        fname = doc_spec["file_name"]
        dtype = doc_spec["doc_type"]
        raw_pages: List[str] = doc_spec["pages"]
        target_path = sample_dir / fname

        if dtype == "pdf":
            pdf_bytes = build_minimal_pdf(raw_pages)
            target_path.write_bytes(pdf_bytes)
            parsed = parse_document(fname, pdf_bytes, doc_type_hint="pdf")
            all_pages.extend(parsed)
        else:
            # Join with explicit page markers so HTML parser preserves exact 2-page structure
            joined_html = "\n\n--- PAGE 1 ---\n\n" + raw_pages[0] + "\n\n--- PAGE 2 ---\n\n" + raw_pages[1]
            target_path.write_text(joined_html, encoding="utf-8")
            parsed = parse_document(fname, joined_html.encode("utf-8"), doc_type_hint="webpage")
            all_pages.extend(parsed)

    return all_pages


def seed_default_corpus(
    vector_store: CampusVectorStore,
    sample_dir: Path,
    chunk_size: int = 550,
    chunk_overlap: int = 120,
    force_rebuild: bool = False,
) -> int:
    """Seed the vector store with the default campus corpus if empty or when `force_rebuild=True`."""
    pages = ensure_sample_campus_docs(sample_dir)
    if vector_store.chunks and not force_rebuild:
        return len(vector_store.chunks)

    vector_store.clear()
    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return vector_store.add_chunks(chunks, replace_existing_file=True)

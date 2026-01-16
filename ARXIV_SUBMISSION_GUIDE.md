# arXiv Submission Guide for Paper 1

## Paper Title
**"Emergent Specialization in Multi-Agent Systems: Competition as the Source of Diversity"**

**Author:** Yuhao Li (University of Pennsylvania)

---

## ✅ Pre-Submission Checklist

### Files Ready for Submission

| File | Status | Notes |
|------|--------|-------|
| `paper/main.tex` | ✅ Ready | Main paper (~10 pages + appendix) |
| `paper/neurips_2024.sty` | ✅ Ready | NeurIPS style file |
| `paper/references.bib` | ✅ Ready | 18 references |
| `paper/appendix_workflow_v2.tex` | ✅ Ready | Detailed algorithm appendix |
| `paper/figures/fig1_cross_domain_si.pdf` | ✅ Ready | Main results figure |
| `paper/figures/fig2_lambda_ablation.pdf` | ✅ Ready | λ ablation figure |
| `paper/figures/fig3_method_specialization.pdf` | ✅ Ready | Method specialization figure |
| `paper/figures/fig4_marl_comparison.pdf` | ✅ Ready | MARL baseline comparison |
| `paper/figures/fig5_summary_heatmap.pdf` | ✅ Ready | Summary heatmap |

### Content Verification

- [x] Abstract: Clear, ~250 words, states contributions
- [x] Introduction: Motivates problem, states contributions
- [x] Related Work: Covers MARL, QD, ecology, ensemble methods
- [x] Method: NichePopulation algorithm with pseudocode
- [x] Theory: 3 propositions with proofs
- [x] Experiments: 6 real-world domains, 30 trials each
- [x] Results: Tables with effect sizes, p-values
- [x] Limitations: Honest discussion of 6 limitations
- [x] Conclusion: Summarizes findings
- [x] Appendix: Detailed algorithm walkthrough

---

## 📋 Submission Steps

### Step 1: Create arXiv Account (if needed)
1. Go to https://arxiv.org
2. Click "Register" and create account
3. Use your UPenn email (li88@sas.upenn.edu) for institutional credibility

### Step 2: Check Endorsement
- First-time submitters to `cs.LG` or `cs.MA` may need endorsement
- UPenn affiliation should provide automatic endorsement
- If not, ask a colleague who has published on arXiv

### Step 3: Prepare Submission Package
Run this command to create the submission zip:

```bash
cd /Users/yuhaoli/code/MAS_For_Finance/emergent_specialization/paper
mkdir -p arxiv_submission
cp main.tex arxiv_submission/
cp neurips_2024.sty arxiv_submission/
cp references.bib arxiv_submission/
cp appendix_workflow_v2.tex arxiv_submission/
cp -r figures arxiv_submission/
cd arxiv_submission
zip -r ../arxiv_submission.zip *
cd ..
echo "Created arxiv_submission.zip"
```

### Step 4: Submit to arXiv
1. Go to https://arxiv.org/submit
2. Select category: **cs.LG** (Machine Learning) or **cs.MA** (Multi-Agent Systems)
3. Upload `arxiv_submission.zip`
4. Fill in metadata:
   - **Title:** Emergent Specialization in Multi-Agent Systems: Competition as the Source of Diversity
   - **Authors:** Yuhao Li
   - **Abstract:** (copy from main.tex)
   - **Comments:** 10 pages, 5 figures, code available at https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems
5. Choose license: **CC BY 4.0** (recommended for maximum visibility)
6. Submit

### Step 5: Wait for Processing
- arXiv typically processes submissions within 1-2 business days
- You'll receive an arXiv ID (e.g., arXiv:2501.xxxxx)
- Paper will appear on arXiv after next announcement (Mon-Fri)

---

## 🎯 Recommended arXiv Categories

**Primary:** `cs.LG` (Machine Learning)
- Most relevant audience
- Highest visibility for ML papers

**Cross-list (optional):**
- `cs.MA` (Multi-Agent Systems)
- `cs.AI` (Artificial Intelligence)
- `cs.NE` (Neural and Evolutionary Computing)

---

## 📝 Key Claims in Paper

| Claim | Evidence |
|-------|----------|
| Competition induces specialization | Mean SI = 0.747, Cohen's d > 20 |
| Competition alone is sufficient | SI = 0.329 at λ=0 (vs 0.127 random) |
| Outperforms MARL | 4.3× higher SI than QMIX/MAPPO/IQL |
| Method division of labor | 87% method coverage, +26.5% performance |
| Theoretical foundation | 3 propositions with proofs |
| Real-world validation | 6 domains, 145K+ records |

---

## ⚠️ Common Issues to Avoid

1. **Missing style file**: neurips_2024.sty must be included ✅ Fixed
2. **Figure paths**: Use relative paths (figures/fig1...) ✅ Verified
3. **Bibliography**: Ensure all citations resolve ✅ 18 refs complete
4. **Compilation errors**: Test with pdflatex locally first
5. **Overfull boxes**: Check for text spilling into margins

---

## 🔗 Related Links

- **Code repository:** https://github.com/HowardLiYH/Emergent-Specialization-in-Multi-Agent-Systems
- **arXiv submission portal:** https://arxiv.org/submit
- **arXiv help:** https://info.arxiv.org/help/submit.html

---

## 📅 Timeline

| Date | Action |
|------|--------|
| Today | Review this guide, prepare files |
| Day 1 | Create arXiv account (if needed), upload submission |
| Day 2-3 | arXiv processing |
| Day 3-4 | Paper appears on arXiv with ID |
| After arXiv | Can cite as "Li, 2025, arXiv:2501.xxxxx" |

---

## 💡 After Submission

1. **Share the arXiv link** on:
   - Twitter/X
   - LinkedIn
   - Relevant subreddits (r/MachineLearning)
   - Research mailing lists

2. **Update Paper 2** to cite this arXiv version:
   ```latex
   \cite{li2025emergent}
   ```

3. **Submit to conference** (NeurIPS, ICML, ICLR) with arXiv as backup

---

*Guide created: January 16, 2026*

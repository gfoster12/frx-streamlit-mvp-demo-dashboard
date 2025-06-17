import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
import tempfile
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Foster Rx: Interactive System Architecture")

# --- Show logo at the top ---
st.image("FullLogo.jpg", use_container_width=False, width=400)

modules = [
    "Data Ingestion",
    "Digital Twin NAMs",
    "Translational Research Engine",
    "LLM Clinical Assistant",
    "Embedded Chat",
    "Matchmaking Engine",
    "External Partners",
]

colors = [
    "#b2f0e9", "#59d2fe", "#2cb3d8", "#1e7fc2",
    "#1565a2", "#0b3d91", "#062c5d"
]

links = [
    (0, 1, 10, "Patient Data & Omics"),
    (1, 2, 8, "Synthetic Cohorts"),
    (2, 3, 6, "Insights & Targets"),
    (3, 4, 4, "Clinical Q&A"),
    (2, 5, 3, "Asset Recommendations"),
    (5, 6, 3, "Partner Matching"),
    (4, 6, 2, "Collaboration"),
]

link_colors = [
    "rgba(41,182,246,0.5)", "rgba(44,179,216,0.5)", "rgba(30,127,194,0.5)",
    "rgba(21,101,162,0.5)", "rgba(11,61,145,0.5)", "rgba(6,44,93,0.5)",
    "rgba(41,182,246,0.3)"
]

node = dict(
    pad=20, thickness=30,
    line=dict(color="black", width=0.5),
    label=modules, color=colors,
)
link = dict(
    source=[s for s, t, v, l in links],
    target=[t for s, t, v, l in links],
    value=[v for s, t, v, l in links],
    label=[l for s, t, v, l in links],
    color=link_colors,
)

fig = go.Figure(go.Sankey(node=node, link=link))
fig.update_layout(
    margin=dict(l=30, r=30, t=30, b=30), font=dict(size=16),
    hovermode="x", plot_bgcolor="#16191c", paper_bgcolor="#16191c",
    title_font_color="#e0e0e0",
    title="System Architecture: Data & Workflow Flows"
)

st.plotly_chart(fig, use_container_width=True)

selected_module = st.radio(
    "Explore a module's journey:",
    [
        "Digital Twin NAMs",
        "Translational Research Engine",
        "LLM Clinical Assistant",
        "Embedded Chat",
        "Matchmaking Engine"
    ],
    horizontal=True,
    index=0
)

if "dtwin_step" not in st.session_state:
    st.session_state.dtwin_step = 1
if "trans_step" not in st.session_state:
    st.session_state.trans_step = 1
if "llm_step" not in st.session_state:
    st.session_state.llm_step = 1
if "match_step" not in st.session_state:
    st.session_state.match_step = 1
if "chat_step" not in st.session_state:
    st.session_state.chat_step = 1
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Export helpers ---
def generate_digital_twin_excel():
    x = np.linspace(0, 12, 100)
    efficacy = 1 - np.exp(-x / 4)
    toxicity = 0.1 + 0.05 * np.sin(x)
    df_sim = pd.DataFrame({'Week': x, 'Efficacy': efficacy, 'Toxicity': toxicity})
    df_cohort = pd.DataFrame({
        "Cohort": ["A", "B", "C", "D"],
        "Evidence Score": np.random.randint(20, 60, 4)
    })
    fig, ax = plt.subplots()
    ax.plot(x, efficacy, label="Efficacy")
    ax.plot(x, toxicity, label="Toxicity", linestyle='--')
    ax.set_xlabel("Simulation Time (weeks)")
    ax.set_ylabel("Response")
    ax.legend()
    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format='png')
    plt.close(fig)
    plot_buf.seek(0)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_sim.to_excel(writer, sheet_name='Sim_Curves', index=False)
        df_cohort.to_excel(writer, sheet_name='Cohorts', index=False)
        workbook = writer.book
        worksheet = workbook.add_worksheet('Visualization')
        writer.sheets['Visualization'] = worksheet
        worksheet.insert_image('B2', 'plot.png', {'image_data': plot_buf})
    output.seek(0)
    return output

def generate_translational_csv():
    stages = ["Preclinical", "Phase I", "Phase II", "Phase III"]
    your_counts = [3, 2, 1, 0]
    comp_counts = [6, 4, 2, 1]
    df_pipeline = pd.DataFrame({
        "Stage": stages,
        "Your Asset": your_counts,
        "Competitors": comp_counts,
    })
    endpoints = ["PFS", "OS", "ORR", "QoL"]
    freq = [0.9, 0.8, 0.6, 0.4]
    regtag = ["FDA", "FDA", "EUA", "None"]
    df_endpoints = pd.DataFrame({
        "Endpoint": endpoints,
        "Usage Freq": freq,
        "Reg Tag": regtag
    })
    buf = io.StringIO()
    buf.write("Pipeline Overlap\n")
    df_pipeline.to_csv(buf, index=False)
    buf.write("\nEndpoint Usage\n")
    df_endpoints.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def generate_matchmaking_pdf():
    partners = ["Sponsor A", "CRO B", "Pharma C"]
    rationale = ["High pathway synergy", "KOL overlap", "Unmet need in territory"]
    fit_scores = [89, 81, 76]
    radar_metrics = ["LoS", "IP Depth", "KOL Score", "Feasibility", "BD Synergy"]
    radar_values = [80, 65, 90, 70, 60]
    report_text = (
        "Matchmaking Simulation Report\n\n"
        "Purpose: This report summarizes the results of the matchmaking simulation, which aims to identify "
        "optimal partners for asset commercialization based on asset profile, development phase, and opportunity fit.\n\n"
        "Top Partner Matches:\n"
    )
    for i, partner in enumerate(partners):
        report_text += f"  - {partner}: {rationale[i]} (Fit Score: {fit_scores[i]})\n"
    report_text += (
        "\nOpportunity Fit Scoring:\n"
        "Multi-factor scoring was applied across key dimensions: LoS, IP Depth, KOL Score, Feasibility, and BD Synergy.\n"
        f"Radar Chart Values: {dict(zip(radar_metrics, radar_values))}\n"
        "\nConclusion: The simulation suggests immediate outreach to Sponsor A and CRO B, "
        "given their high partnership synergy and strategic alignment.\n"
    )
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in report_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.ln(8)
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(0, 10, "Partner Matches Table", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(40, 8, "Partner", border=1)
    pdf.cell(70, 8, "Rationale", border=1)
    pdf.cell(30, 8, "Fit Score", border=1)
    pdf.ln()
    for i in range(len(partners)):
        pdf.cell(40, 8, partners[i], border=1)
        pdf.cell(70, 8, rationale[i], border=1)
        pdf.cell(30, 8, str(fit_scores[i]), border=1)
        pdf.ln()
    import math
    values = radar_values + [radar_values[0]]
    labels = radar_metrics + [radar_metrics[0]]
    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    fig, ax = plt.subplots(subplot_kw={'polar': True})
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    plt.tight_layout()
    radar_bytes = io.BytesIO()
    plt.savefig(radar_bytes, format='PNG')
    plt.close(fig)
    radar_bytes.seek(0)
    # Write radar image to temporary file for FPDF
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        tmp_img.write(radar_bytes.read())
        tmp_img.flush()
        img_path = tmp_img.name
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Opportunity Fit Radar Chart", ln=True)
    pdf.image(img_path, x=40, y=30, w=120)
    out_pdf = pdf.output(dest='S').encode('latin1')
    return out_pdf

# --- Digital Twin NAMs Animated Flow ---
if selected_module == "Digital Twin NAMs":
    st.header("Digital Twin NAMs: Simulation Journey")
    if st.session_state.dtwin_step == 1:
        st.subheader("1. Enter Compound and Select Multi-Omics Data")
        with st.form("dtwin_input_form"):
            compound = st.selectbox("Select Compound", ["Drug 28", "Drug 42", "Drug 7"], key="dtwin_compound")
            population = st.selectbox("Target Population", ["Adults 13-65", "Pediatrics", "Seniors"], key="dtwin_population")
            dosage = st.slider("Dosage Level (mg/kg)", 1, 100, 12, key="dtwin_dosage")
            duration = st.selectbox("Duration", ["4 weeks", "12 weeks", "24 weeks"], key="dtwin_duration")
            omics_data = st.selectbox("Multi-Omics Data", ["Synthetic Transcriptomics", "Synthetic Proteomics", "Upload File..."], key="dtwin_omics")
            submitted = st.form_submit_button("Start Simulation ▶️")
            if submitted:
                st.session_state.dtwin_step = 2
                st.rerun()
    elif st.session_state.dtwin_step == 2:
        st.subheader("2. Simulation Running...")
        with st.spinner("Simulating efficacy, toxicity, and indications..."):
            import time
            time.sleep(2)
        st.success("Simulation complete!")
        st.session_state.dtwin_step = 3
        st.rerun()
    elif st.session_state.dtwin_step == 3:
        st.subheader("3. Simulation Results & Outputs")
        col1, col2 = st.columns(2)
        x = np.linspace(0, 12, 100)
        y = 1 - np.exp(-x / 4)
        tox = 0.1 + 0.05 * np.sin(x)
        df_sim = pd.DataFrame({'Week': x, 'Efficacy': y, 'Toxicity': tox})
        df_cohort = pd.DataFrame({
            "Cohort": ["A", "B", "C", "D"],
            "Evidence Score": np.random.randint(20, 60, 4)
        })
        with col1:
            st.markdown("**Efficacy and Toxicity Curve**")
            fig_, ax = plt.subplots()
            ax.plot(x, y, label="Efficacy")
            ax.plot(x, tox, label="Toxicity", linestyle='--')
            ax.set_xlabel("Simulation Time (weeks)")
            ax.set_ylabel("Response")
            ax.legend()
            st.pyplot(fig_)
            st.markdown("**Real-World Evidence (Fake Data)**")
            st.bar_chart(df_cohort.set_index("Cohort"))
        with col2:
            st.markdown("**Parallel Indications (Feasibility vs. Market Size)**")
            feas = np.random.uniform(0.5, 1, 10)
            market = np.random.uniform(50, 200, 10)
            fig2, ax2 = plt.subplots()
            ax2.scatter(market, feas, c='teal', s=80)
            ax2.set_xlabel("Market Size (M USD)")
            ax2.set_ylabel("Feasibility Score")
            st.pyplot(fig2)
            st.markdown("**Organ Toxicity Profile**")
            organs = ["Liver", "Kidney", "Cardiac"]
            risk = np.random.choice(["Low", "Moderate", "High"], 3, p=[0.6, 0.3, 0.1])
            colors_ = {"Low": "green", "Moderate": "orange", "High": "red"}
            fig3, ax3 = plt.subplots()
            ax3.bar(organs, [1 if r == "Low" else 2 if r == "Moderate" else 3 for r in risk],
                    color=[colors_[r] for r in risk])
            ax3.set_yticks([1,2,3])
            ax3.set_yticklabels(["Low", "Moderate", "High"])
            st.pyplot(fig3)
        st.markdown("---")
        colA, colB = st.columns([2,1])
        with colA:
            st.info("**Collaboration Insight:**\n\nBased on your simulation, Sponsor XYZ has a matching pipeline in oncology. [Contact Sponsor XYZ](#)")
        with colB:
            excel_bytes = generate_digital_twin_excel()
            st.download_button(
                label="Export Results (Excel, with data and visuals)",
                data=excel_bytes,
                file_name="digital_twin_simulation.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.caption("The file contains simulation data, cohort scores, and a visualization sheet.")
        st.markdown("#### Simulation Assistant Chat")
        st.chat_message("user").write("Which indication offers the highest ROI?")
        st.chat_message("assistant").write("Indication 3 (Oncology) shows the highest predicted ROI based on simulated feasibility and market estimates.")
        st.chat_message("user").write("Can you explain this real-world evidence trend?")
        st.chat_message("assistant").write("The real-world evidence chart reflects higher outcomes in Cohort D, likely due to enriched patient selection in the simulation settings.")
        st.button("Restart Simulation", on_click=lambda: st.session_state.update({"dtwin_step": 1}))

# --- Translational Research Engine Animated Flow ---
elif selected_module == "Translational Research Engine":
    st.header("Translational Research & Competitive Intelligence: Animated User Flow")
    if st.session_state.trans_step == 1:
        st.subheader("1. Enter Asset and Target Information")
        with st.form("trans_input_form"):
            asset = st.text_input("Asset Name", "Asset 101")
            target_class = st.selectbox("Target Class", ["Kinase", "Ion Channel", "GPCR", "Other"])
            indication = st.text_input("Indication", "Oncology")
            pathway = st.text_input("Molecular Pathway", "PI3K/AKT/mTOR")
            gene = st.text_input("Gene Target", "PIK3CA")
            disease = st.text_input("Disease Prevalence", "1.2%")
            submitted = st.form_submit_button("Run Competitive Analysis ▶️")
            if submitted:
                st.session_state.trans_step = 2
                st.rerun()
    elif st.session_state.trans_step == 2:
        st.subheader("2. Competitive Analysis & Indication Recommendation Running...")
        with st.spinner("Running pipeline overlap, clustering, endpoint mining, LoS prediction..."):
            import time
            time.sleep(2)
        st.success("Analysis complete!")
        st.session_state.trans_step = 3
        st.rerun()
    elif st.session_state.trans_step == 3:
        st.subheader("3. Results: Competitive, Indication, Endpoint, LoS, and IP Insights")
        col1, col2 = st.columns(2)
        stages = ["Preclinical", "Phase I", "Phase II", "Phase III"]
        your_counts = np.array([3, 2, 1, 0])
        comp_counts = np.array([6, 4, 2, 1])
        with col1:
            st.markdown("**Competitive Score & Pipeline Overlap**")
            fig_, ax = plt.subplots()
            width = 0.35
            x_ = np.arange(len(stages))
            ax.bar(x_ - width/2, your_counts, width, label="Your Asset")
            ax.bar(x_ + width/2, comp_counts, width, label="Competitors")
            ax.set_xticks(x_)
            ax.set_xticklabels(stages)
            ax.set_ylabel("Number of Assets")
            ax.set_title("Pipeline Overlap")
            ax.legend()
            st.pyplot(fig_)
            st.markdown("**Indication Recommendation (AI Ranked)**")
            recs = ["Oncology", "Neurology", "Immunology"]
            score = np.sort(np.random.rand(3))[::-1]
            df = pd.DataFrame({"Indication": recs, "Score": np.round(score,2)})
            st.dataframe(df, hide_index=True)
        with col2:
            st.markdown("**Endpoint Selection & Validation**")
            endpoints = ["PFS", "OS", "ORR", "QoL"]
            freq = [0.9, 0.8, 0.6, 0.4]
            df2 = pd.DataFrame({"Endpoint": endpoints, "Usage Freq": freq, "Reg Tag": ["FDA", "FDA", "EUA", "None"]})
            st.dataframe(df2, hide_index=True)
            st.markdown("**Likelihood of Success (LoS) Prediction**")
            los = np.round(np.random.uniform(0.15, 0.71),2)
            feat_imp = {"Preclinical data":0.28, "Trial design":0.54, "Compound meta":0.18}
            st.metric("LoS (%)", los*100)
            st.bar_chart(pd.Series(feat_imp))
            st.markdown("**Publication & IP Insights**")
            st.write("• Top Publications: [PMID: 123456](#), [PMID: 7891011](#)")
            st.write("• Patents: US1234567, US2345678")
        st.markdown("---")
        colA, colB = st.columns([2,1])
        with colA:
            st.info("**Acceptance Criteria:**\n- Recommendation & endpoint results include citations.\n- LoS engine outputs are exportable.\n- Response time <4 seconds for all modules.")
        with colB:
            csv_buf = generate_translational_csv()
            st.download_button(
                label="Export Report (CSV, simulation data)",
                data=csv_buf.getvalue(),
                file_name="translational_research_simulation.csv",
                mime="text/csv"
            )
            st.caption("The CSV contains pipeline overlap and endpoint usage simulation results.")
        st.button("Restart Analysis", on_click=lambda: st.session_state.update({"trans_step": 1}))

# --- LLM Clinical Assistant Animated Flow ---
elif selected_module == "LLM Clinical Assistant":
    st.header("LLM-Powered Clinical Assistant: Animated User Flow")
    if st.session_state.llm_step == 1:
        st.subheader("1. Enter Clinical Query")
        with st.form("llm_input_form"):
            user_query = st.text_area(
                "Enter your domain-specific query (e.g. 'Design trial for pediatric Pompe disease')",
                "Design trial for pediatric Pompe disease"
            )
            submitted = st.form_submit_button("Ask Assistant ▶️")
            if submitted:
                st.session_state.llm_query_val = user_query
                st.session_state.llm_step = 2
                st.rerun()
    elif st.session_state.llm_step == 2:
        st.subheader("2. Query Processing")
        with st.spinner("Retrieving evidence, parsing eligibility, generating output..."):
            import time
            time.sleep(2)
        st.success("Result ready!")
        st.session_state.llm_step = 3
        st.rerun()
    elif st.session_state.llm_step == 3:
        st.subheader("3. Output: Structured Assistant Response")
        st.markdown("**Domain-Specific Query Handling**\n- Sources: PubMed, ClinicalTrials.gov, Orphanet, NIH GARD\n- RAG pipeline using BioBERT/ClinicalBERT embeddings")
        st.markdown("**Trial Design Recommendation**")
        st.json({
            "Phase": "II",
            "Endpoints": ["PFS", "ORR", "QoL"],
            "Duration": "24 months",
            "Justification": "Phase II chosen for new population, endpoints per FDA precedence."
        })
        st.markdown("**Eligibility Parsing**")
        st.table(pd.DataFrame({
            "Inclusion": ["Genetically confirmed", "Age 2-16"],
            "Exclusion": ["Prior enzyme therapy"]
        }, index=[1,2]))
        st.markdown("**Citations**")
        st.write("- [PMID: 123456 PubMed](#)")
        st.write("- [ClinicalTrials.gov: NCT01234567](#)")
        st.markdown("**Non-Functional**")
        st.write("- LLM fine-tuned on LoRA/PEFT using HuggingFace\n- Grounding rate >80%\n- Output includes at least one citation per query")
        st.button("Restart", on_click=lambda: st.session_state.update({"llm_step": 1}))

# --- Embedded Chat Widget (Contextual Assistant) Animated Flow ---
elif selected_module == "Embedded Chat":
    st.header("Embedded Chat Widget (Contextual Assistant): Animated User Flow")
    if st.session_state.chat_step == 1:
        st.subheader("1. Universal Access Button")
        st.info("• Floating chat icon appears on all screens (bottom-right corner)\n• Sticky element across all modules")
        if st.button("Open Chat ▶️", key="open_chat"):
            st.session_state.chat_step = 2
            st.rerun()
    elif st.session_state.chat_step == 2:
        st.subheader("2. Contextual Assistant Logic & Prompt Suggestions")
        st.info("• Context detected (module/sub-module)\n• LLM responses adjust based on context\n• Sample prompts, tooltips, and summaries offered based on task")
        prompt = st.selectbox("Prompt Suggestions", [
            "Explain toxicity risk",
            "Summarize simulation results",
            "How do I export data?",
            "What does 'LoS' mean here?"
        ])
        if st.button("Continue to Chat", key="continue_chat"):
            st.session_state.selected_prompt = prompt
            st.session_state.chat_step = 3
            st.rerun()
    elif st.session_state.chat_step == 3:
        st.subheader("3. Chat Interaction & Session Integration")
        st.info("• Chat assistant integrates with LLM and Digital Twin modules.\n• Can call internal APIs for data.\n• Session history is searchable (FAISS/Pinecone).")
        chat_input = st.text_input("Type your message:", st.session_state.get("selected_prompt", ""))
        if st.button("Send", key="send_chat"):
            user_msg = chat_input
            bot_msg = ""
            if "toxicity" in user_msg.lower():
                bot_msg = "Toxicity risk for the selected agent is low, based on simulation data."
            elif "export" in user_msg.lower():
                bot_msg = "You can export all session data using the 'Export Results' button in the module."
            elif "los" in user_msg.lower():
                bot_msg = "'LoS' stands for Likelihood of Success, calculated using historical trial outcomes."
            elif "summarize" in user_msg.lower() or "result" in user_msg.lower():
                bot_msg = "The simulation returned high efficacy and moderate safety signals for the selected cohort."
            elif user_msg.strip() == "":
                bot_msg = "Please enter a question."
            else:
                bot_msg = "I'm here to help! Please ask a module-specific question."
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            st.session_state.chat_history.append({"role": "assistant", "content": bot_msg})
            st.rerun()
        for msg in st.session_state.chat_history[-6:]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
        if st.button("Show Session History & Retrieval", key="show_history"):
            st.session_state.chat_step = 4
            st.rerun()
        if st.button("Restart Chat", key="restart_chat"):
            st.session_state.chat_history = []
            st.session_state.chat_step = 1
            st.rerun()
    elif st.session_state.chat_step == 4:
        st.subheader("4. Session History & Retrieval")
        st.info("• All previous chat interactions are stored as vector embeddings (FAISS/Pinecone).\n• Searchable by keyword or date.")
        search = st.text_input("Search session history by keyword:", "")
        results = []
        if search:
            results = [m for m in st.session_state.chat_history if search.lower() in m["content"].lower()]
        else:
            results = st.session_state.chat_history[-10:]
        for msg in results:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
        if st.button("Back to Chat", key="back_to_chat"):
            st.session_state.chat_step = 3
            st.rerun()
        if st.button("Restart Widget", key="restart_widget"):
            st.session_state.chat_history = []
            st.session_state.chat_step = 1
            st.rerun()

# --- Matchmaking Engine Animated Flow ---
elif selected_module == "Matchmaking Engine":
    st.header("Biopharma Collaboration & Commercialization Engine: Animated User Flow")
    if st.session_state.match_step == 1:
        st.subheader("1. Enter Asset and Matching Criteria")
        with st.form("match_input_form"):
            asset_profile = st.text_input("Asset Profile", "First-in-class kinase inhibitor")
            phase = st.selectbox("Development Phase", ["Preclinical", "Phase I", "Phase II", "Phase III"])
            target = st.text_input("Target", "EGFR")
            modality = st.selectbox("Modality", ["Small Molecule", "Biologic", "Other"])
            submitted = st.form_submit_button("Find Partners ▶️")
            if submitted:
                st.session_state.match_step = 2
                st.rerun()
    elif st.session_state.match_step == 2:
        st.subheader("2. Running Partner Matching & Opportunity Scoring...")
        with st.spinner("Running graph-based sponsor matching, scoring opportunity fit, and trend mining..."):
            import time
            time.sleep(2)
        st.success("Results ready!")
        st.session_state.match_step = 3
        st.rerun()
    elif st.session_state.match_step == 3:
        st.subheader("3. Outputs: Partner Matches, Score, Trends, and Collaboration Hub")
        col1, col2 = st.columns(2)
        partners = ["Sponsor A", "CRO B", "Pharma C"]
        rationale = ["High pathway synergy", "KOL overlap", "Unmet need in territory"]
        fit_scores = [89, 81, 76]
        radar_metrics = ["LoS", "IP Depth", "KOL Score", "Feasibility", "BD Synergy"]
        radar_values = [80, 65, 90, 70, 60]
        with col1:
            st.markdown("**Top Partner Matches (Graph-based)**")
            st.dataframe(pd.DataFrame({"Partner": partners, "Rationale": rationale, "Fit Score": fit_scores}))
            st.markdown("**Opportunity Fit Score (Radar Chart)**")
            import plotly.express as px
            radar_df = pd.DataFrame(dict(
                r=radar_values,
                theta=radar_metrics
            ))
            radar_df = pd.concat([radar_df, radar_df.iloc[[0]]])
            fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True)
            fig_radar.update_traces(fill='toself')
            st.plotly_chart(fig_radar, use_container_width=True)
        with col2:
            st.markdown("**Collaboration Hub**")
            st.write("- Submit asset to partner")
            st.write("- Request meeting")
            st.write("- Share pitch summary")
            st.write("- Track proposals via audit dashboard")
            st.markdown("**Regulatory & Reimbursement Trends**")
            st.write("• HTA trend: [NICE, HAS, GBA] - Favorable for rare oncology assets.")
            pdf_bytes = generate_matchmaking_pdf()
            st.download_button(
                label="Export Narrative PDF Report",
                data=pdf_bytes,
                file_name="matchmaking_simulation_report.pdf",
                mime="application/pdf"
            )
            st.caption("The PDF contains a written explanation, a table, and a radar chart visualization.")
        st.button("Restart", on_click=lambda: st.session_state.update({"match_step": 1}))

st.caption("Choose a module above to explore its interactive journey. Node click simulation is best achieved with these radio buttons in Streamlit.")

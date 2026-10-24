"""
This code generates the user interface and accepts new inputs.
Enter "streamlit run datacenter_gui.py" in the terminal to run.
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from solver import default_params, make_tasks, solve_milp, greedy_schedule, compute_metrics
from solver import fig_schedule, fig_greedy_vs_milp, fig_gantt, table_tasks, table_hourly_power

# ==========================================
# INIT
# ==========================================
st.set_page_config(layout="wide")

# Ensure the session state is initialized
if "P" not in st.session_state:
    st.session_state.P = default_params()
if "tasks" not in st.session_state:
    st.session_state.tasks = make_tasks(st.session_state.P)

P = st.session_state.P

# ==========================================
# INTERFACE
# ==========================================
st.title("Data Center Energy Optimization Dashboard")

with st.sidebar:
    # --- RESET EVERYTHING BUTTON ---
    if st.button("Reset to Default Settings & Tasks", use_container_width=True):
        st.session_state.P = default_params()
        st.session_state.tasks = make_tasks(st.session_state.P)
        st.rerun()

    st.divider()

    # --- ADD NEW TASK FORM ---
    with st.container():
        st.subheader("New Task Input")
        with st.form(key="task_input_form", clear_on_submit=True):
            s = st.number_input("Arrival Time Step", min_value=0, max_value=23, value=8)
            e = st.number_input("Deadline Time Step", min_value=0, max_value=23, value=12)
            cnt = st.number_input("Number of Tasks", min_value=1, step=1)
            uw = st.number_input("kWh per task", min_value=1, max_value=5000)
            priority = st.selectbox("Priority", ["MEDIUM (finish within 4h)", "HIGH (finish within 1h)", "LOW (finish by end of day)"])
            
            if st.form_submit_button("Add Task"):
                if s < e:
                    # Quick physics check before adding
                    required_power = (cnt * uw) / (e - s) / P["eta"]
                    if required_power > P["C"]:
                        st.error(f"Too much work! Needs {required_power:.0f} kW/hr, but max is {P['C']:.0f}.")
                    else:
                        # Add a new custom task to session_state
                        new_id = max([t["id"] for t in st.session_state.tasks], default=0) + 1
                        urg = "HIGH" if "HIGH" in priority else ("LOW" if "LOW" in priority else "MED")
                        
                        st.session_state.tasks.append({
                            "id": new_id,
                            "a": int(s),
                            "d": int(e),
                            #"w": float(cnt * uw) / (e - s),
                            "w": float(cnt * uw),
                            "u": urg
                        })
                        st.rerun()
                else:
                    st.error("Deadline must be > Arrival")

    # --- EDIT PARAMETERS ---
    with st.container():
        st.subheader("Data Center Capacity")
        with st.form(key="data_center_form"):
            new_C = st.number_input("Maximum compute power (kWh per step)", value=float(P["C"]), min_value=1.0)
            new_eta = st.number_input("Power-to-compute efficiency", value=float(P["eta"]), min_value=0.01, max_value=1.0)
            if st.form_submit_button("Edit"):
                st.session_state.P["C"] = new_C
                st.session_state.P["eta"] = new_eta
                st.rerun()

    with st.container():
        st.subheader("Battery Parameters")
        with st.form(key="battery_form"):
            new_B_max = st.number_input("Maximum battery capacity (kWh)", value=float(P["B_max"]), min_value=1.0)  
            new_B_init = st.number_input("Initial battery state of charge (kWh)", value=float(P["B_init"]), min_value=0.0, max_value=float(P["B_max"]))  
            new_P_batt_max = st.number_input("Maximum battery charge/discharge power (kW)", value=float(P["P_batt_max"]), min_value=1.0)  
            new_eta_ch = st.number_input("Charging efficiency", value=float(P["eta_ch"]), min_value=0.01, max_value=1.0) 
            new_eta_dis = st.number_input("Discharging efficiency", value=float(P["eta_dis"]), min_value=0.01, max_value=1.0) 
            if st.form_submit_button("Edit"):
                st.session_state.P["B_max"] = new_B_max
                st.session_state.P["B_init"] = new_B_init
                st.session_state.P["P_batt_max"] = new_P_batt_max
                st.session_state.P["eta_ch"] = new_eta_ch
                st.session_state.P["eta_dis"] = new_eta_dis
                st.rerun()

    # --- MANAGE & DELETE TASKS ---
    with st.expander("Manage & Delete Tasks", expanded=False):
        st.write("Click a row and click the trash can icon to delete a task.")
        
        friendly_names = {
            "id": "Task ID",
            "a": "Arrival Hour",
            "d": "Deadline Hour",
            "w": "Power (kW)",
            "u": "Urgency"
        }
        
        # Convert to DataFrame and rename columns
        df_edit = pd.DataFrame(st.session_state.tasks)
        df_edit = df_edit[["id", "a", "d", "w", "u"]]
        df_edit = df_edit.rename(columns=friendly_names)

        # Display editable table (with delete checkboxes enabled)
        edited_df = st.data_editor(
            df_edit, 
            key="task_editor", 
            num_rows="dynamic", 
            hide_index=True,
            use_container_width=True
        )
        
        # If the user deleted rows or edited values, update session state
        if not edited_df.equals(df_edit):
            reverse_names = {"Task ID": "id", "Arrival Hour": "a", "Deadline Hour": "d", "Power (kW)": "w", "Urgency": "u"}
            st.session_state.tasks = edited_df.rename(columns=reverse_names).to_dict('records')

            st.rerun()

# ==========================================
# RUN THE MATH MODEL
# ==========================================
tasks = st.session_state.tasks

if len(tasks) > 0:
    res = solve_milp(P, tasks)

    if res is None:
        st.error("Infeasible schedule! You asked for more compute than the data center can physically handle before the deadline. Try deleting some tasks or lowering 'kWh per task'.")
    else:
        # Only draw charts if the solve is successful
        m_milp = compute_metrics(P, tasks, res["sched"], res["P_g"], res["P_t"])
        g_sched, g_pg, g_pt = greedy_schedule(P, tasks)
        m_greedy = compute_metrics(P, tasks, g_sched, g_pg, g_pt)

        schedule = fig_schedule(res, P)
        gantt = fig_gantt(res, tasks, P)
        greedy_vs_milp = fig_greedy_vs_milp(m_milp, m_greedy)

        df_tasks = table_tasks(res, tasks)
        df_hourly_power = table_hourly_power(P, res)

        # ==========================================
        # OPTIMIZATION DISPLAY
        # ==========================================
        tab1, tab2, tab3 = st.tabs(["Computation Schedule", "Power Consumption", "Metrics: Comparison with Greedy Baseline"])

        with tab1:
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.pyplot(gantt)
                plt.close(gantt)
            with col1_2:    
                st.dataframe(df_tasks, use_container_width=True)

        with tab2:
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.pyplot(schedule)
                plt.close(schedule)
            with col2_2:
                st.dataframe(df_hourly_power, use_container_width=True)

        with tab3:
            st.pyplot(greedy_vs_milp)
            plt.close(greedy_vs_milp)
else:
    st.info("No tasks scheduled. Add a task in the sidebar.")
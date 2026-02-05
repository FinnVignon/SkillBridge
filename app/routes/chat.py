from __future__ import annotations

from typing import Dict, List, TypedDict

from flask import Blueprint, current_app, render_template, request, session

from services.store import Store, get_store

chat_bp = Blueprint("chat", __name__)


class HistoryItem(TypedDict):
    role: str
    text: str


class ChatState(TypedDict):
    step: str
    role: str
    job_title: str
    domaine: str
    competences: str
    niveau: str
    rome: str
    formacode: str
    history: List[HistoryItem]
    last_user: str


QUESTIONS_BY_ROLE = {
    "apprenant": [
        ("job_title", "Quel intitulé de poste vise l’étudiant ? (ex: assistant comptable ; technicien maintenance)"),
        ("domaine", "Quel domaine d’activité ? (ex: comptabilité ; froid industriel ; robotique)"),
        ("competences", "Quelles compétences clés doivent apparaître ? (ex: maintenance système ; diagnostic ; soudure)"),
        ("niveau", "Niveau RNCP souhaité (le résultat ne dépassera pas ce niveau) ? (ex: 3 ou niveau 3)"),
        ("rome", "Code/libellé ROME ciblé ? (optionnel, laissez vide si inconnu)"),
        ("formacode", "Formacode/libellé ciblé ? (optionnel, laissez vide si inconnu)"),
    ],
    "ecole": [
        ("job_title", "Quel intitulé ou formation vise l’offre ? (ex: développeur web ; assistant RH)"),
        ("domaine", "Quel domaine d’activité couvre l’offre ?"),
        ("competences", "Quelles missions/compétences sont attendues ?"),
        ("niveau", "Niveau RNCP visé par l’école ?"),
    ],
    "employeur": [
        ("job_title", "Quel poste l’entreprise souhaite pourvoir ?"),
        ("domaine", "Quel domaine d’activité pour l’entreprise ?"),
        ("competences", "Quelles compétences clés attendues ?"),
        ("niveau", "Niveau RNCP souhaité ?"),
    ],
}


def init_state() -> ChatState:
    return {
        "step": "0",
        "role": "apprenant",
        "job_title": "",
        "domaine": "",
        "competences": "",
        "niveau": "",
        "rome": "",
        "formacode": "",
        "history": [],
        "last_user": "",
    }


def build_query_by_cat(state: ChatState, store: Store) -> Dict[str, str]:
    return {
        "job_title": state.get("job_title", "").strip(),
        "domaine": state.get("domaine", "").strip(),
        "niveau": store.normalize_niveau_input(state.get("niveau", "")),
        "competences": state.get("competences", "").strip(),
        "rome": state.get("rome", "").strip(),
        "formacode": state.get("formacode", "").strip(),
    }


@chat_bp.route("/chat", methods=["GET", "POST"])
def chat():
    store = get_store(current_app)
    if "state" not in session:
        session["state"] = init_state()

    state: ChatState = dict(session["state"])
    role = request.args.get("role") or state.get("role", "apprenant")
    if role not in QUESTIONS_BY_ROLE:
        role = "apprenant"
    state["role"] = role
    questions = QUESTIONS_BY_ROLE[role]
    results = []
    keywords = []
    done = False
    history = list(state.get("history", []))
    last_user = state.get("last_user", "")

    if request.method == "POST":
        action = request.form.get("action", "send")
        user_input = request.form.get("query", "").strip()
        step_idx = int(state.get("step", "0"))

        if action == "restart":
            state = init_state()
            session["state"] = state
            return render_template("home.html")

        if step_idx >= len(questions):
            state = init_state()
            step_idx = 0
            history = []
            last_user = ""
            state["role"] = role

        if step_idx < len(questions):
            if action == "skip":
                key, _ = questions[step_idx]
                state[key] = ";"
                state["step"] = str(step_idx + 1)
                last_user = ""
            elif user_input:
                key, _ = questions[step_idx]
                state[key] = user_input
                state["step"] = str(step_idx + 1)
                history.append({"role": "user", "text": user_input})
                last_user = user_input

        if int(state["step"]) >= len(questions):
            done = True
            history.append({"role": "bot", "text": "Voici les meilleurs résultats selon vos critères."})

        state["history"] = history
        state["last_user"] = last_user
        session["state"] = state

    step_idx = int(state.get("step", "0"))
    if step_idx < 0 or step_idx > len(questions):
        state = init_state()
        session["state"] = state
        step_idx = 0
        done = False
        history = []
        last_user = ""
    done = done or step_idx >= len(questions)
    current_question = ""
    if not done and step_idx < len(questions):
        current_question = questions[step_idx][1]
        if not history or history[-1].get("text") != current_question:
            history.append({"role": "bot", "text": current_question})
    elif done and not history:
        history.append({"role": "bot", "text": "Voici les meilleurs résultats selon vos critères."})
    elif not history:
        history.append({"role": "bot", "text": "Posez-moi une nouvelle recherche si besoin."})

    if state.get("history") != history:
        state["history"] = history
        session["state"] = state

    analysis = {}
    if done:
        query_by_cat = build_query_by_cat(state, store)
        results, keywords = store.search(query_by_cat)
        analysis = store.build_role_analysis(role, query_by_cat, results)

    return render_template(
        "index.html",
        history=history,
        current_question=current_question,
        last_user=last_user,
        results=results,
        keywords=keywords,
        step=int(state.get("step", "0")),
        total_steps=len(questions),
        done=done,
        default_question=questions[0][1] if questions else "",
        role=role,
        analysis=analysis,
    )

"""AgentDojo harness: the Week-0 baseline, and the seam the defense will slot into.

WHAT THIS PACKAGE IS
--------------------
AgentDojo is an external prompt-injection benchmark. It drives an agent through
tool-using tasks, injects attacker text into tool outputs, and reports two
numbers: task *utility* (did the agent do the user's job) and *attack success
rate* (did the injected instruction hijack a tool call). We run it here against a
plain agent-under-test with no Aegis defense, so the number is an honest floor:
this is what an undefended agent scores, and every later defended run is measured
as the distance from it.

THE TWO PROVIDER PATHS
----------------------
* ``openai-compat`` (the default, primary path) uses AgentDojo's own,
  battle-tested ``OpenAILLM`` against an OpenAI-compatible endpoint - Groq's
  ``https://api.groq.com/openai/v1`` with ``openai/gpt-oss-120b`` by default,
  which is the exact configuration the recorded Week-0 baseline ran on (capable
  enough to solve the tasks AND measurably hijackable; NVIDIA's small llama stays
  supported via flags but is not a valid security baseline - see the runner's
  module docstring). It needs no protocol surgery, so the baseline rests on stock
  plumbing (see :mod:`evals.agentdojo.openai_llm`).
* ``gemini`` (secondary, experimental) uses the one piece that is not
  off-the-shelf: AgentDojo ships a ``GoogleLLM``, but it predates Gemini 3.x and
  breaks on it - 3.x requires the per-call ``thought_signature`` the model emits
  on a tool-call turn to be echoed back verbatim on the next turn, and the stock
  element rebuilds bare ``functionCall`` parts that drop it, so the second turn of
  any tool loop 400s. :mod:`evals.agentdojo.gemini_llm` fixes exactly that and
  nothing else (see its module docstring for the mechanism).

THE SEAM FOR THE DEFENSE
------------------------
The runner builds the pipeline with ``defense=None`` today. That is the single
place the Aegis defense will later be introduced - as an AgentDojo defense name
or a wrapping pipeline element - without touching the LLM element or the metric
plumbing. Keeping the baseline path clean now is what makes that swap a one-line
change later.
"""

from __future__ import annotations

from evals.agentdojo.gemini_llm import Gemini3LLM, Gemini3LLMError, build_gemini_llm
from evals.agentdojo.openai_llm import (
    CountingOpenAILLM,
    OpenAICompatLLMError,
    build_openai_compat_llm,
)

# One builder per provider, exported symmetrically: each owns reading its own key
# from the variable the caller names, so the runner dispatches on provider and
# nothing else.
__all__ = [
    "CountingOpenAILLM",
    "Gemini3LLM",
    "Gemini3LLMError",
    "OpenAICompatLLMError",
    "build_gemini_llm",
    "build_openai_compat_llm",
]

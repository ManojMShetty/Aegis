"""Framework adapters - the code a library cannot write on your behalf.

:mod:`aegis.middleware` deliberately knows nothing about messages, runtimes or
tool objects. Something still has to: find the tool results in whatever shape a
framework uses, put the guarded spans back where they came from, read a pending
call out of that framework's own object, and phrase a refusal the way its loop
expects. That is what an adapter is, and it is the only part of this project that
takes a dependency on somebody else's agent framework.

WHY THESE ARE NOT IMPORTED HERE
-------------------------------
Each adapter needs its framework installed, and none of them is needed to use the
library. Importing them eagerly would make ``langgraph`` a hard requirement of
``import aegis.adapters`` and, through anything that touched it, of the middleware
itself - the exact defect this project already shipped once with ``httpx`` and the
Gemini provider, and only found by installing the wheel somewhere clean.

So each adapter is imported from its own module and the extra is named in the
error when it is missing::

    pip install aegis-rag[langchain]
    from aegis.adapters.langgraph import AegisToolNode

WHAT LIVES HERE AND WHAT MUST NOT
---------------------------------
Adapters may import a framework; that is their entire purpose. Nothing under
:mod:`aegis.middleware`, :mod:`aegis.security` or :mod:`aegis.domain` may, and a
test asserts it - because the claim being defended is that the judgments and the
runtime stand on their own, not that this package has no framework code anywhere.
"""

from __future__ import annotations

__all__: list[str] = []

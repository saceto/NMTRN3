"""NVIDIA Build / NIM inference path for the desktop agent."""

from __future__ import annotations

from server.agent import NemotronAgent


class NvidiaInferenceAgent(NemotronAgent):
    """Provider-specific class for NVIDIA-hosted OpenAI-compatible inference."""

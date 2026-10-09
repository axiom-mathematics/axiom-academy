// Optional attempt sync. Disabled unless config.js sets enabled: true.
// No keys ship in the repository.

window.AxiomSync = {
  async submit(payload) {
    const config = window.AXIOM_SYNC || { enabled: false };
    if (!config.enabled) {
      return { synced: false, reason: "Sync is off. Download the attempt file and send it to Steve." };
    }
    if (config.provider === "supabase") {
      if (!config.supabaseUrl || !config.supabaseAnonKey) {
        throw new Error("Supabase URL and anon key are missing.");
      }
      const response = await fetch(`${config.supabaseUrl.replace(/\/$/, "")}/rest/v1/${config.table || "axiom_attempts"}`, {
        method: "POST",
        headers: {
          apikey: config.supabaseAnonKey,
          Authorization: `Bearer ${config.supabaseAnonKey}`,
          "Content-Type": "application/json",
          Prefer: "return=minimal",
        },
        body: JSON.stringify({ payload }),
      });
      if (!response.ok) throw new Error(await response.text());
      return { synced: true };
    }
    if (config.provider === "firebase") {
      if (!config.projectId) throw new Error("Firebase projectId is missing.");
      const response = await fetch(
        `https://firestore.googleapis.com/v1/projects/${config.projectId}/databases/(default)/documents/${config.collection || "axiom_attempts"}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fields: {
              payload: { stringValue: JSON.stringify(payload) },
              createdAt: { timestampValue: new Date().toISOString() },
            },
          }),
        },
      );
      if (!response.ok) throw new Error(await response.text());
      return { synced: true };
    }
    throw new Error("Set provider to supabase or firebase, or leave sync disabled.");
  },
};

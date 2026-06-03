import express from "express";
import cors from "cors";
import { askAssistant } from "./Assistant.js";

const app = express();

app.use(cors());
app.use(express.json());

app.post("/chat", async (req, res) => {
  try {
    const { message , mood, history, sessionId} = req.body;
    console.log("Received:", { message, mood , history, sessionId});
    const reply = await askAssistant(message, mood, history, sessionId);
    res.json({ reply });
  } catch (error) {
    res.status(500).json({ error: error.message});
  }
});

app.listen(3000, () => {
  console.log("Server running on port 3000");
});

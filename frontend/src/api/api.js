import axios from "axios";

export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export const runScript = async (scriptName, params = {}) => {
  // Send params directly as the POST body
  const res = await axios.post(`${API_BASE}/run/${scriptName}`, params);
  return res.data;
};

export const uploadFile = async (scriptName, file) => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await axios.post(`${API_BASE}/run/${scriptName}`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return res.data;
};
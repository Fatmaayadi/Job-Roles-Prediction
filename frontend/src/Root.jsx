import { useState } from "react";
import Auth from "./Auth.jsx";
import App from "./App.jsx";

export default function Root() {
  const [user, setUser] = useState(null);

  if (!user) {
    return <Auth onAuthenticated={(data) => setUser(data)} />;
  }

  return <App user={user} onLogout={() => setUser(null)} />;
}
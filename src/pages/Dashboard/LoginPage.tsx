import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./LoginPage.css";

interface LoginPageProps {
  onLogin: () => void;
}

export const LoginPage: React.FC<LoginPageProps> = ({ onLogin }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();

    // Fake auth for now
    if (email && password) {
      localStorage.setItem("authUser", JSON.stringify({ email }));
      onLogin();
      navigate("../");
    } else {
      alert("Please enter valid email and password");
    }
  };

  return (
    <div className="login-container">
      <div className="login-animated-bg" aria-hidden="true">
        <span className="blob blob-1" />
        <span className="blob blob-2" />
        <span className="blob blob-3" />
      </div>

      <div className="login-grid">
        <aside className="brand-side">
          <div className="brand-inner">
            <div className="brand-badge">EngEX</div>
            <h1 className="brand-title">EngEX Exhibition</h1>
            <p className="brand-subtitle">
              Where innovation meets experience. Join exhibitors and visitors
              for an immersive showcase of technology, products, and ideas.
            </p>
            <ul className="brand-points">
              <li>Smart booth management</li>
              <li>Real-time analytics</li>
              <li>Interactive attendee journey</li>
            </ul>
            <a className="brand-cta" href="#" onClick={(e) => e.preventDefault()}>
              Explore Highlights
            </a>
          </div>
        </aside>

        <div className="login-card">
          <div className="login-card-header">
            <h2 className="login-title">Welcome back</h2>
            <p className="login-subtitle">Sign in to manage your EngEX booth</p>
          </div>

          <form onSubmit={handleLogin} className="login-form">
            <div className="form-group">
              <label className="form-label">Email</label>
              <div className="input-with-icon">
                <span className="input-icon" aria-hidden="true">
                  {/* Mail icon */}
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 6h16a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="m22 8-10 6L2 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </span>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="form-input"
                  placeholder="you@company.com"
                  required
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Password</label>
              <div className="input-with-icon">
                <span className="input-icon" aria-hidden="true">
                  {/* Lock icon */}
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="4" y="10" width="16" height="10" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M8 10V7a4 4 0 1 1 8 0v3" stroke="currentColor" strokeWidth="1.5"/>
                  </svg>
                </span>
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="form-input"
                  placeholder="Enter your password"
                  required
                />
                <button
                  type="button"
                  className="toggle-password"
                  onClick={() => setShowPassword((s) => !s)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? "Hide" : "Show"}
                </button>
              </div>
            </div>

            <div className="form-row">
              <label className="checkbox">
                <input type="checkbox" />
                <span>Remember me</span>
              </label>
              <a className="link" href="#" onClick={(e) => e.preventDefault()}>
                Forgot password?
              </a>
            </div>

            <div className="actions">
              <button type="submit" className="login-button">Login</button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => navigate("../register")}
              >
                Create account
              </button>
            </div>
          </form>

          <p className="disclaimer">
            By continuing, you agree to the <a href="#" onClick={(e) => e.preventDefault()} className="link">Terms</a> and
            <a href="#" onClick={(e) => e.preventDefault()} className="link"> Privacy Policy</a>.
          </p>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;

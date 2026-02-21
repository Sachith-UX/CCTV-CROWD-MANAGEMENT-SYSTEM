import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "./RegisterPage.css";

const RegisterPage: React.FC = () => {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [company, setCompany] = useState("");
  const [agree, setAgree] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!agree) {
      alert("Please accept the terms and privacy policy.");
      return;
    }
    if (password !== confirmPassword) {
      alert("Passwords do not match");
      return;
    }
    if (!email || !password || !firstName) {
      alert("Please fill all required fields");
      return;
    }

    // Placeholder for real registration logic.
    const user = { firstName, lastName, email, company };
    localStorage.setItem("registeredUser", JSON.stringify(user));
    // Navigate back to login
    navigate("../login");
  };

  return (
    <div className="auth-container">
      <div className="auth-animated-bg" aria-hidden="true">
        <span className="blob blob-1" />
        <span className="blob blob-2" />
        <span className="blob blob-3" />
      </div>

      <div className="auth-grid">
        <aside className="brand-side">
          <div className="brand-inner">
            <div className="brand-badge">EngEX</div>
            <h1 className="brand-title">Create your EngEX account</h1>
            <p className="brand-subtitle">
              Register to showcase, manage, and analyze your exhibition presence.
            </p>
            <ul className="brand-points">
              <li>Booth tools for teams</li>
              <li>Lead capture and insights</li>
              <li>Seamless event operations</li>
            </ul>
          </div>
        </aside>

        <div className="auth-card">
          <div className="auth-card-header">
            <h2 className="auth-title">Create account</h2>
            <p className="auth-subtitle">Join EngEX Exhibition</p>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            <div className="form-row two">
              <div className="form-group">
                <label className="form-label">First name</label>
                <input
                  className="form-input"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  placeholder="Jane"
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">Last name</label>
                <input
                  className="form-input"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  placeholder="Doe"
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Company / Organization</label>
              <input
                className="form-input"
                value={company}
                onChange={(e) => setCompany(e.target.value)}
                placeholder="Acme Corp"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Email</label>
              <input
                type="email"
                className="form-input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@company.com"
                required
              />
            </div>

            <div className="form-row two">
              <div className="form-group">
                <label className="form-label">Password</label>
                <div className="input-with-toggle">
                  <input
                    type={showPassword ? "text" : "password"}
                    className="form-input"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Create a password"
                    required
                    minLength={6}
                  />
                  <button
                    type="button"
                    className="toggle-password"
                    onClick={() => setShowPassword((s) => !s)}
                  >
                    {showPassword ? "Hide" : "Show"}
                  </button>
                </div>
              </div>
              <div className="form-group">
                <label className="form-label">Confirm password</label>
                <input
                  type="password"
                  className="form-input"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Repeat password"
                  required
                  minLength={6}
                />
              </div>
            </div>

            <label className="checkbox">
              <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
              <span>
                I agree to the <a href="#" onClick={(e) => e.preventDefault()} className="link">Terms</a> and
                <a href="#" onClick={(e) => e.preventDefault()} className="link"> Privacy Policy</a>.
              </span>
            </label>

            <button type="submit" className="primary-button">Create account</button>
            <p className="switch">
              Already have an account? <Link className="link" to="../login">Log in</Link>
            </p>
          </form>
        </div>
      </div>
    </div>
  );
};

export default RegisterPage;

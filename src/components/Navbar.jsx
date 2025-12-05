import './Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-title">Flood Prediction</div>
      <div className="navbar-controls">
        <button className="navbar-btn minimize">_</button>
        <button className="navbar-btn maximize">□</button>
        <button className="navbar-btn close">×</button>
      </div>
    </nav>
  );
}

export default Navbar;

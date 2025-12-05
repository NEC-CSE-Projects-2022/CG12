import './App.css';
import Navbar from './components/Navbar';
import HeaderCard from './components/HeaderCard';
import Tabs from './components/Tabs';
import Footer from './components/Footer';

function App() {
  return (
    <div className="app">
      <Navbar />
      <HeaderCard />
      <div className="main-content">
        <div className="content-area">
          <Tabs />
        </div>
      </div>
      <Footer />
    </div>
  );
}

export default App;

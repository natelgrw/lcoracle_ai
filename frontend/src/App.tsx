import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import AmaxModule from './pages/modules/AmaxModule';
import AskcosModule from './pages/modules/AskcosModule';
import RetinaModule from './pages/modules/RetinaModule';
import PeakProphetModule from './pages/modules/PeakProphetModule';
import GradienceModule from './pages/modules/GradienceModule';
import Layout from './components/Layout';

import ScrollToTop from './components/ScrollToTop';

function App() {
  return (
    <Router>
      <ScrollToTop />
      <Layout>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/amax" element={<AmaxModule />} />
          <Route path="/askcos" element={<AskcosModule />} />
          <Route path="/retina" element={<RetinaModule />} />
          <Route path="/peak-prophet" element={<PeakProphetModule />} />
          <Route path="/gradience" element={<GradienceModule />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;

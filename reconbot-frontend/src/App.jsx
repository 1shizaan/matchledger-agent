// // ~/reconbot/reconbot-frontend/src/App.jsx
// import { Routes, Route, Navigate } from 'react-router-dom';
// import { useAuth } from './contexts/AuthContext';
// import Dashboard from './pages/Dashboard';
// import Login from './pages/Login';
// import Register from './components/Auth/Register';
// import ForgotPassword from './pages/ForgotPassword';
// import ResetPassword from './pages/ResetPassword';

// // ProtectedRoute component - Fixed to use correct property names
// const ProtectedRoute = ({ children }) => {
//   const { user, loading } = useAuth(); // Using 'user' and 'loading' from your AuthContext

//   console.log('ProtectedRoute - user:', user, 'loading:', loading); // Debug log

//   // If still loading auth state, show a loading indicator
//   if (loading) {
//     return (
//       <div className="flex items-center justify-center min-h-screen">
//         <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
//         <span className="ml-2">Loading authentication...</span>
//       </div>
//     );
//   }

//   // If not authenticated (no user) and not loading, redirect to login
//   if (!user) {
//     console.log('ProtectedRoute - No user found, redirecting to login');
//     return <Navigate to="/login" replace />;
//   }

//   console.log('ProtectedRoute - User authenticated, rendering protected content');
//   return children;
// };

// // PublicRoute component - Redirects authenticated users away from login/register pages
// const PublicRoute = ({ children }) => {
//   const { user, loading } = useAuth();

//   console.log('PublicRoute - user:', user, 'loading:', loading); // Debug log

//   // If still loading, show loading indicator
//   if (loading) {
//     return (
//       <div className="flex items-center justify-center min-h-screen">
//         <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
//         <span className="ml-2">Loading...</span>
//       </div>
//     );
//   }

//   // If user is authenticated, redirect to dashboard
//   if (user) {
//     console.log('PublicRoute - User authenticated, redirecting to dashboard');
//     return <Navigate to="/dashboard" replace />;
//   }

//   console.log('PublicRoute - No user, rendering public content');
//   return children;
// };

// function App() {
//   return (
//     <Routes>
//       {/* Public routes - redirect to dashboard if already authenticated */}
//       <Route
//         path="/login"
//         element={
//           <PublicRoute>
//             <Login />
//           </PublicRoute>
//         }
//       />
//       <Route
//         path="/register"
//         element={
//           <PublicRoute>
//             <Register />
//           </PublicRoute>
//         }
//       />
//       <Route path="/forgot-password" element={<ForgotPassword />} />
//       <Route path="/reset-password" element={<ResetPassword />} />

//       {/* Protected routes */}
//       <Route
//         path="/dashboard"
//         element={
//           <ProtectedRoute>
//             <Dashboard />
//           </ProtectedRoute>
//         }
//       />

//       {/* Default route: redirects based on authentication status */}
//       <Route
//         path="/"
//         element={
//           <ProtectedRoute>
//             <Dashboard />
//           </ProtectedRoute>
//         }
//       />

//       {/* Catch-all for undefined routes */}
//       <Route path="*" element={<Navigate to="/" replace />} />
//     </Routes>
//   );
// }

// export default App;

// ~/reconbot/reconbot-frontend/src/App.jsx
import React, { Suspense, lazy, useState, useEffect, createContext, useContext } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './contexts/AuthContext';
import Footer from './components/Footer';

// Create Dark Mode Context
const DarkModeContext = createContext();

export const useDarkMode = () => {
  const context = useContext(DarkModeContext);
  if (!context) {
    throw new Error('useDarkMode must be used within a DarkModeProvider');
  }
  return context;
};

// Lazy load route components
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Login = lazy(() => import('./pages/Login'));
const Register = lazy(() => import('./components/Auth/Register'));
const ForgotPassword = lazy(() => import('./pages/ForgotPassword'));
const ResetPassword = lazy(() => import('./pages/ResetPassword'));

const LoadingSpinner = () => (
  <div className="flex items-center justify-center min-h-screen bg-gray-100 dark:bg-gray-900">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 dark:border-blue-400"></div>
    <span className="ml-2 text-gray-700 dark:text-gray-300">Loading...</span>
  </div>
);

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) return <LoadingSpinner />;
  if (!user) return <Navigate to="/login" replace />;

  return children;
};

const PublicRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) return <LoadingSpinner />;
  if (user) return <Navigate to="/dashboard" replace />;

  return children;
};

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const savedMode = localStorage.getItem('darkMode');
    return savedMode === 'true';
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(prev => !prev);

  const darkModeValue = {
    darkMode,
    toggleDarkMode
  };

  return (
    <DarkModeContext.Provider value={darkModeValue}>
      <div className="min-h-screen flex flex-col bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
        <Suspense fallback={<LoadingSpinner />}>
          <div className="flex-grow">
            <Routes>
              <Route
                path="/login"
                element={
                  <PublicRoute>
                    <Login />
                  </PublicRoute>
                }
              />
              <Route
                path="/register"
                element={
                  <PublicRoute>
                    <Register />
                  </PublicRoute>
                }
              />
              <Route path="/forgot-password" element={<ForgotPassword />} />
              <Route path="/reset-password" element={<ResetPassword />} />

              <Route
                path="/dashboard"
                element={
                  <ProtectedRoute>
                    <Dashboard />
                  </ProtectedRoute>
                }
              />

              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <Dashboard />
                  </ProtectedRoute>
                }
              />

              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
          <Footer />
        </Suspense>
      </div>
    </DarkModeContext.Provider>
  );
}

export default App;
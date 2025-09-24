// src/components/Header.jsx
import React from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { FaMoon, FaSun } from 'react-icons/fa'; // Assuming react-icons is installed

// Added userEmail and logoSrc props
const Header = ({ toggleDarkMode, darkMode, userEmail, onLogout, logoSrc }) => {
  const { logout } = useAuth(); // Keeping useAuth for direct logout if needed, but using onLogout prop
  const navigate = useNavigate();

  const handleLogout = () => {
    onLogout(); // Use the prop function
    navigate('/login');
  };

  return (
    <header className="flex justify-between items-center p-4 bg-white dark:bg-gray-800 shadow-md">
      <div className="flex items-center space-x-2">
        {/* Your logo or app name */}
        {logoSrc && <img src={logoSrc} alt="MatchLedger Logo" className="h-8" />} {/* Use logoSrc prop */}
        <span className="text-xl font-bold text-gray-900 dark:text-white">MatchLedger</span>
      </div>
      <div className="flex items-center space-x-4">
        {/* Dark Mode Toggle */}
        <button
          onClick={toggleDarkMode}
          className="p-2 rounded-full text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
          title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {darkMode ? <FaSun size={20} /> : <FaMoon size={20} />}
        </button>

        {/* User Email (assuming you have access to it from useAuth) */}
        <span className="text-gray-700 dark:text-gray-300 hidden sm:block">
          {userEmail || 'Guest'} {/* Display userEmail prop or 'Guest' */}
        </span>

        <button
          onClick={handleLogout}
          className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500"
        >
          Logout
        </button>
      </div>
    </header>
  );
};

export default Header;
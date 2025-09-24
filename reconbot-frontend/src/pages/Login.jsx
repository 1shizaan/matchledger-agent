// // src/pages/Login.jsx
// import { useState } from 'react';
// import GoogleAuth from '../components/Auth/GoogleLogin';
// import ManualLogin from '../components/Auth/ManualLogin';
// import Register from '../components/Auth/Register';

// const Login = () => {
//   const [isRegisterMode, setIsRegisterMode] = useState(false);

//   return (
//     <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
//       <h1 className="text-3xl font-bold mb-6 text-center">
//         {isRegisterMode ? 'Create Account - ReconBot' : 'Sign in to ReconBot'}
//       </h1>

//       {/* Toggle between Login and Register */}
//       <div className="w-full max-w-sm mb-4">
//         <div className="flex bg-gray-200 rounded-lg p-1">
//           <button
//             onClick={() => setIsRegisterMode(false)}
//             className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
//               !isRegisterMode
//                 ? 'bg-white text-gray-900 shadow-sm'
//                 : 'text-gray-600 hover:text-gray-900'
//             }`}
//           >
//             Sign In
//           </button>
//           <button
//             onClick={() => setIsRegisterMode(true)}
//             className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
//               isRegisterMode
//                 ? 'bg-white text-gray-900 shadow-sm'
//                 : 'text-gray-600 hover:text-gray-900'
//             }`}
//           >
//             Register
//           </button>
//         </div>
//       </div>

//       {/* Conditional rendering based on mode */}
//       {isRegisterMode ? (
//         <Register />
//       ) : (
//         <>
//           {/* Manual Login Component */}
//           <ManualLogin />

//           <div className="relative my-6 w-full max-w-sm">
//             <div className="absolute inset-0 flex items-center">
//               <span className="w-full border-t"></span>
//             </div>
//             <div className="relative flex justify-center text-xs uppercase">
//               <span className="bg-gray-100 px-2 text-muted-foreground">Or continue with</span>
//             </div>
//           </div>

//           {/* Google Login Component */}
//           <GoogleAuth />
//         </>
//       )}

//       {/* Alternative text-based toggle at bottom */}
//       <div className="mt-6 text-center">
//         <p className="text-sm text-gray-600">
//           {isRegisterMode ? (
//             <>
//               Already have an account?{' '}
//               <button
//                 onClick={() => setIsRegisterMode(false)}
//                 className="text-blue-600 hover:underline font-medium"
//               >
//                 Sign in here
//               </button>
//             </>
//           ) : (
//             <>
//               Don't have an account?{' '}
//               <button
//                 onClick={() => setIsRegisterMode(true)}
//                 className="text-blue-600 hover:underline font-medium"
//               >
//                 Create one here
//               </button>
//             </>
//           )}
//         </p>
//       </div>
//     </div>
//   );
// };

// export default Login;

// src/pages/Login.jsx
import { useState } from 'react';
import GoogleAuth from '../components/Auth/GoogleLogin';
import ManualLogin from '../components/Auth/ManualLogin';
import Register from '../components/Auth/Register';

const Login = () => {
  const [isRegisterMode, setIsRegisterMode] = useState(false);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-6 text-center">
        {isRegisterMode ? 'Create Account - MatchLedger AI' : 'Sign in to MatchLedger AI'}
      </h1>

      {/* Toggle between Login and Register */}
      <div className="w-full max-w-sm mb-4">
        <div className="flex bg-gray-200 rounded-lg p-1">
          <button
            onClick={() => setIsRegisterMode(false)}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              !isRegisterMode
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Sign In
          </button>
          <button
            onClick={() => setIsRegisterMode(true)}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              isRegisterMode
                ? 'bg-white text-gray-900 shadow-sm'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Register
          </button>
        </div>
      </div>

      {/* Conditional rendering based on mode */}
      {isRegisterMode ? (
        <Register />
      ) : (
        <>
          {/* Manual Login Component */}
          <ManualLogin />

          <div className="relative my-6 w-full max-w-sm">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t"></span>
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-gray-100 px-2 text-muted-foreground">Or continue with</span>
            </div>
          </div>

          {/* Google Login Component */}
          <GoogleAuth />
        </>
      )}

      {/* Alternative text-based toggle at bottom */}
      <div className="mt-6 text-center">
        <p className="text-sm text-gray-600">
          {isRegisterMode ? (
            <>
              Already have an account?{' '}
              <button
                onClick={() => setIsRegisterMode(false)}
                className="text-blue-600 hover:underline font-medium"
              >
                Sign in here
              </button>
            </>
          ) : (
            <>
              Don't have an account?{' '}
              <button
                onClick={() => setIsRegisterMode(true)}
                className="text-blue-600 hover:underline font-medium"
              >
                Create one here
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
};

export default Login;
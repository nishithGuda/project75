import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
    const location = useLocation();
    const navItems = [  
        {label: "Accounts", path:"/"},
        {label: "Transactions", path:"/transactions"},
        {label: "Transfer", path:"/transfer"},
    ];

  return (
    <nav className='bg-white shadow-md py-4 px-6 flex items-center justify-between'>
      <h1 className="text-xl font-bold text-blue-700">MyBank Portal</h1>
      <ul className="flex space-x-6 text-sm font-medium">
        {navItems.map((item) => (
          <li key={item.path}>
            <Link
              to={item.path}
              data-id={`nav-${item.label.toLowerCase()}`}
              data-type="navigation"
              data-label={`${item.label} Navigation Link`}
              className={`${
                location.pathname === item.path
                  ? "text-blue-600 border-b-2 border-blue-600"
                  : "text-gray-600 hover:text-blue-600"
              } pb-1 transition duration-150`}
            >
              {item.label}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
};

export default Navbar;
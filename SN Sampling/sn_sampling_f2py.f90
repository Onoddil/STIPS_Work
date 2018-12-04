! f2py -c -m sn_sampling_f2py sn_sampling_f2py.f90 --f90flags='-Wall -Wextra -Werror -pedantic -fbacktrace -O0 -g -fcheck=all -fopenmp' -lgomp
! f2py -c -m sn_sampling_f2py sn_sampling_f2py.f90 --f90flags='-O3 -fopenmp' -lgomp
subroutine psf_fit_min(p, x, y, z, o_inv_sq, nitem, fun, jac, lenp, lenx, leny)
    implicit none
    integer, intent(in) :: nitem, lenp, lenx, leny
    double precision, intent(in) :: p(lenp), x(lenx), y(leny), z(lenx, leny), o_inv_sq(lenx, leny)
    double precision, intent(out) :: fun, jac(lenp)
    integer :: i, j, k, k_, nset
    double precision :: c, s, mux, muy, pi, f, dx, dy, common_terms
    double precision, allocatable :: g_minus_c(:), f_parts(:)

    pi = 3.141592653589793
    ! p should be sets of (mux, muy, sigma, c) which should be referenced directly
    nset = lenp / nitem ! division of two integers in fortran is the same as a // b in python
    allocate(g_minus_c(nset))
    allocate(f_parts(nset))
    fun = 0.0d0
    jac(:) = 0.0d0
!$OMP PARALLEL DO DEFAULT(NONE) COLLAPSE(2) PRIVATE(i, j, k, f, f_parts, g_minus_c, k_, mux, muy, s, c, dx, dy, &
!$OMP& common_terms) SHARED(nset, nitem, p, x, y, pi, z, o_inv_sq, lenx, leny) REDUCTION(+: fun, jac)
    do i = 1, leny
        do j = 1, lenx
            f_parts(:) = 0.0d0
            g_minus_c(:) = 0.0d0
            do k = 1, nset
                k_ = nitem*(k-1) + 1
                mux = p(k_)
                muy = p(k_ + 1)
                s = p(k_ + 2)
                c = p(k_ + 3)
                dx = x(j) - mux
                dy = y(i) - muy
                g_minus_c(k) = 1.0d0 / (2.0d0 * pi * s) * exp(-0.5 * (dx**2 + dy**2) / s**2)
                f_parts(k) = c * g_minus_c(k)
            end do
            f = sum(f_parts)
            common_terms = 2 * (f - z(j, i)) * o_inv_sq(j, i)
            do k = 1, nset
                k_ = nitem*(k-1) + 1
                mux = p(k_)
                muy = p(k_ + 1)
                s = p(k_ + 2)
                c = p(k_ + 3)
                dx = x(j) - mux
                dy = y(i) - muy
                ! the function f in these sums is not sum_j f_ij above, but the individual gaussian component for this
                !parameter, which are referenced through f_parts individually
                jac(k_) = jac(k_) + common_terms * f_parts(k) * dx / s**2
                jac(k_ + 1) = jac(k_ + 1) + common_terms * f_parts(k) * dy / s**2
                jac(k_ + 2) = jac(k_ + 2) + common_terms * f_parts(k) * ((dx**2 + dy**2) / s**2 - 1) / s
                jac(k_ + 3) = jac(k_ + 3) + common_terms * g_minus_c(k)
            end do
            fun = fun + (f - z(j, i))**2 * o_inv_sq(j, i)
        end do
    end do
!$OMP END PARALLEL DO

end subroutine psf_fit_min

subroutine psf_fit_hess(p, x, y, z, o_inv_sq, nitem, hess, lenp, lenx, leny)
    implicit none
    integer, intent(in) :: nitem, lenp, lenx, leny
    double precision, intent(in) :: p(lenp), x(lenx), y(leny), z(lenx, leny), o_inv_sq(lenx, leny)
    double precision, intent(out) :: hess(lenp, lenp)
    integer :: i, j, k, l, kp, lp, nset, k_, l_
    double precision :: c, s, mux, muy, pi, f, common_terms, dx, dy, q
    double precision, allocatable :: g_minus_c(:), dfda(:, :), f_parts(:)

    pi = 3.141592653589793
    ! p should be sets of (mux, muy, sigma, c) which should be referenced directly
    nset = lenp / nitem ! division of two integers in fortran is the same as a // b in python
    hess(:, :) = 0.0d0
    allocate(g_minus_c(nset))
    allocate(dfda(nitem, nset))
    allocate(f_parts(nset))
!$OMP PARALLEL DO DEFAULT(NONE) COLLAPSE(2) PRIVATE(i, j, k, l, k_, l_, dfda, f, mux, muy, s, c, dx, dy, g_minus_c, &
!$OMP& f_parts, common_terms, kp, lp, q) SHARED(nitem, nset, lenx, leny, p, x, y, z, o_inv_sq, pi) REDUCTION(+: hess)
    do i = 1, leny
        do j = 1, lenx
            dfda(:, :) = 0.0d0
            f = 0.0d0
            do k = 1, nset
                k_ = nitem*(k-1) + 1
                mux = p(k_)
                muy = p(k_ + 1)
                s = p(k_ + 2)
                c = p(k_ + 3)
                dx = x(j) - mux
                dy = y(i) - muy
                g_minus_c(k) = 1.0d0 / (2.0d0 * pi * s) * exp(-0.5 * (dx**2 + dy**2) / s**2)
                f_parts(k) = c * g_minus_c(k)
            end do
            f = sum(f_parts)
            ! dfda is just dfda, NOT dFda, which would otherwise add sum_i 2 (sum_j f_ij - z_i) dfda into the mix, so
            ! there's no common terms factor here; the f in dfda is not sum_j f_ij but a specific f_k, referenced as
            ! f_parts
            do k = 1, nset
                k_ = nitem*(k-1) + 1
                mux = p(k_)
                muy = p(k_ + 1)
                s = p(k_ + 2)
                c = p(k_ + 3)
                dx = x(j) - mux
                dy = y(i) - muy
                dfda(1, k) = f_parts(k) * dx / s**2
                dfda(2, k) = f_parts(k) * dy / s**2
                dfda(3, k) = f_parts(k) * ((dx**2 + dy**2) / s**2 - 1) / s
                dfda(4, k) = g_minus_c(k)
            end do
            do k = 1, nset
                k_ = nitem*(k-1) + 1
                do l = 1, nset
                    l_ = nitem*(l-1) + 1
                    common_terms = 2.0d0 * o_inv_sq(j, i)
                    do kp = 0, nitem - 1
                        do lp = 0, nitem - 1
                            ! dfda must be referenced as *p+1 because of the zero-index offsets below...
                            hess(l_+lp, k_+kp) = hess(l_+lp, k_+kp) + common_terms * dfda(kp+1, k) * dfda(lp+1, l)
                        end do
                    end do
                    if (k == l) then
                        mux = p(k_)
                        muy = p(k_ + 1)
                        s = p(k_ + 2)
                        c = p(k_ + 3)
                        dx = x(j) - mux
                        dy = y(i) - muy
                        ! this f is the sum_j f_ij, so keep as the 'proper' f
                        common_terms = 2.0d0 * (f - z(j, i)) * o_inv_sq(j, i)
                        hess(l_, k_) = hess(l_, k_) + common_terms * f_parts(k) / s**2 * ((dx / s)**2 - 1)
                        
                        q = common_terms * f_parts(k) * dx * dy / s**4
                        hess(l_, k_ + 1) = hess(l_, k_ + 1) + q
                        hess(l_ + 1, k_) = hess(l_ + 1, k_) + q
                        
                        q = common_terms * f_parts(k) * dx * ((dx**2 + dy**2) / s**2 - 3) / s**3
                        hess(l_, k_ + 2) = hess(l_, k_ + 2) + q
                        hess(l_ + 2, k_) = hess(l_ + 2, k_) + q
                        
                        q = common_terms * g_minus_c(k) * dx / s**2
                        hess(l_, k_ + 3) = hess(l_, k_ + 3) + q
                        hess(l_ + 3, k_) = hess(l_ + 3, k_) + q
                        
                        hess(l_ + 1, k_ + 1) = hess(l_ + 1, k_ + 1) + common_terms * f_parts(k)/s**2 * ((dy/s)**2 - 1)
                        
                        q = common_terms * f_parts(k) * dy * ((dx**2 + dy**2) / s**2 - 3) / s**3
                        hess(l_ + 1, k_ + 2) = hess(l_ + 1, k_ + 2) + q
                        hess(l_ + 2, k_ + 1) = hess(l_ + 2, k_ + 1) + q
                        
                        q = common_terms * g_minus_c(k) * dy / s**2
                        hess(l_ + 1, k_ + 3) = hess(l_ + 1, k_ + 3) + q
                        hess(l_ + 3, k_ + 1) = hess(l_ + 3, k_ + 1) + q

                        q = common_terms*f_parts(k)/s**2 * (1+((dx**2 + dy**2) / s**2 - 1)**2 - 3*(dx**2 + dy**2)/s**2)
                        hess(l_ + 2, k_ + 2) = hess(l_ + 2, k_ + 2) + q

                        q = common_terms * g_minus_c(k) * ((dx**2 + dy**2) / s**2 - 1) / s
                        hess(l_ + 2, k_ + 3) = hess(l_ + 2, k_ + 3) + q
                        hess(l_ + 3, k_ + 2) = hess(l_ + 3, k_ + 2) + q
                        ! no 3, 3 as d2fdc2 = 0
                    end if
                end do
            end do
        end do
    end do

end subroutine psf_fit_hess